from dataclasses import dataclass
import torch

@dataclass
class SISAConfig:
    n_shards: int = 5
    shard_size: int = 12000
    hidden_size: int = 256
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"