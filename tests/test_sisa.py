from src.unlearning.sisa import SISAConfig, SISAManager
import torch
import time
import pytest

def test_sisa_unlearning():
    torch.manual_seed(42)
    
    # Initialize configuration
    config = SISAConfig(
        n_shards=5,
        shard_size=12000,
        hidden_size=256,
        learning_rate=0.001,
        batch_size=64,
        epochs=3  # Reduced for testing
    )
    
    print("Initializing SISA Manager...")
    sisa = SISAManager(config)
    
    # Train all shards
    print("\nStarting training...")
    start_time = time.time()
    sisa.train_all_shards()
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Initial evaluation
    print("\nEvaluating initial model...")
    _, test_data = sisa.load_data()
    metrics = sisa.evaluate(test_data)
    print(f"Initial test accuracy: {metrics['accuracy']:.2f}%")
    
    # Test unlearning
    print("\nTesting unlearning...")
    sample_to_unlearn = 42  
    start_time = time.time()
    shard_id = sisa.unlearn_sample(sample_to_unlearn)
    unlearning_time = time.time() - start_time
    
    if shard_id is not None:
        print(f"Unlearned sample {sample_to_unlearn} from shard {shard_id}")
        print(f"Unlearning completed in {unlearning_time:.2f} seconds")
    else:
        print(f"Sample {sample_to_unlearn} not found in any shard")
    
    # Final evaluation
    print("\nEvaluating after unlearning...")
    metrics = sisa.evaluate(test_data)
    print(f"Final test accuracy: {metrics['accuracy']:.2f}%")
    
    sisa.save_state("models/sisa_state.pkl")

if __name__ == "__main__":
    test_sisa_unlearning()