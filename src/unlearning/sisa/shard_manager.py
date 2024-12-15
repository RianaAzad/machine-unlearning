import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import pickle
from pathlib import Path

@dataclass
class SISAConfig:
    n_shards: int = 5
    shard_size: int = 12000
    hidden_size: int = 256
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class MNISTNet(nn.Module):
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 10),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class SISAManager:
    def __init__(self, config: SISAConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.models: Dict[int, MNISTNet] = {}
        self.data_maps: Dict[int, List[int]] = {}
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> Tuple[Dataset, Dataset]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('./data', train=False, transform=transform)
        return train_data, test_data
        
    def create_shards(self, dataset: Dataset) -> Dict[int, Subset]:
        indices = torch.randperm(len(dataset))
        shards = {}
        
        for shard_id in range(self.config.n_shards):
            start_idx = shard_id * self.config.shard_size
            end_idx = start_idx + self.config.shard_size
            shard_indices = indices[start_idx:end_idx]
            shards[shard_id] = Subset(dataset, shard_indices)
            self.data_maps[shard_id] = shard_indices.tolist()
            
        return shards
        
    def train_shard(self, shard_id: int, shard_data: Subset):
        model = MNISTNet(self.config.hidden_size).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.NLLLoss()
        
        train_loader = DataLoader(
            shard_data, 
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        model.train()
        for epoch in range(self.config.epochs):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            avg_loss = running_loss / len(train_loader)
            self.logger.info(f'Shard {shard_id}, Epoch {epoch+1}/{self.config.epochs}, Loss: {avg_loss:.4f}')
            
        self.models[shard_id] = model
        
    def train_all_shards(self):
        train_data, _ = self.load_data()
        shards = self.create_shards(train_data)
        
        for shard_id, shard_data in shards.items():
            self.logger.info(f'Training shard {shard_id}')
            self.train_shard(shard_id, shard_data)
            
    def unlearn_sample(self, sample_idx: int) -> Optional[int]:
        """Remove a sample's influence by identifying and retraining its shard"""
        for shard_id, indices in self.data_maps.items():
            if sample_idx in indices:
                # Remove sample from shard's data
                shard_indices = [idx for idx in indices if idx != sample_idx]
                train_data, _ = self.load_data()
                new_shard = Subset(train_data, shard_indices)
                
                # Retrain shard
                self.logger.info(f'Unlearning sample {sample_idx} from shard {shard_id}')
                self.train_shard(shard_id, new_shard)
                self.data_maps[shard_id] = shard_indices
                
                return shard_id
        return None
        
    def evaluate(self, test_data: Dataset) -> Dict[str, float]:
        test_loader = DataLoader(test_data, batch_size=self.config.batch_size)
        ensemble_correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                ensemble_output = torch.zeros(len(data), 10).to(self.device)
                
                # Aggregate predictions from all shards
                for model in self.models.values():
                    model.eval()
                    output = model(data)
                    ensemble_output += torch.exp(output)
                
                ensemble_output /= len(self.models)
                pred = ensemble_output.argmax(dim=1)
                ensemble_correct += pred.eq(target).sum().item()
                total += len(data)
        
        accuracy = 100. * ensemble_correct / total
        self.logger.info(f'Test Accuracy: {accuracy:.2f}%')
        return {"accuracy": accuracy}
    
    def save_state(self, path: str):
        state = {
            'data_maps': self.data_maps,
            'config': self.config
        }
        model_states = {shard_id: model.state_dict() 
                       for shard_id, model in self.models.items()}
        state['model_states'] = model_states
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        self.logger.info(f'Saved SISA state to {path}')
    
    def load_state(self, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.data_maps = state['data_maps']
        self.config = state['config']
        
        for shard_id, model_state in state['model_states'].items():
            model = MNISTNet(self.config.hidden_size).to(self.device)
            model.load_state_dict(model_state)
            self.models[shard_id] = model
        self.logger.info(f'Loaded SISA state from {path}')