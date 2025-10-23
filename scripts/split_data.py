"""
Split the medical NER dataset into train, validation, and test sets.
- Train: 80% (2,400 samples) - for fine-tuning
- Validation: 10% (300 samples) - for monitoring during training (W&B)
- Test: 10% (300 samples) - for final evaluation after training
"""

import json
import random
from sklearn.model_selection import train_test_split

def main():
    print("Loading dataset...")
    
    # Load data
    with open('../data/both_rel_instruct_all.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Total samples loaded: {len(data)}")
    
    # First split: 80% train, 20% temp (for val + test)
    random.seed(42)
    train_data, temp_data = train_test_split(
        data, 
        test_size=0.2,  # 20% for val + test
        random_state=42,
        shuffle=True
    )
    
    # Second split: split the 20% into 10% val, 10% test
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,  # 50% of 20% = 10% of total
        random_state=42,
        shuffle=True
    )
    
    # Save train set
    with open('../data/train.jsonl', 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    # Save validation set (used during training)
    with open('../data/validation.jsonl', 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    # Save test set (used only for final evaluation)
    with open('../data/test.jsonl', 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n✓ Successfully split dataset:")
    print(f"  - Train samples: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  - Validation samples: {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")
    print(f"  - Test samples: {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")
    
    print(f"\n📊 Dataset usage:")
    print(f"  - Train: Used for fine-tuning the model")
    print(f"  - Validation: Used during training to monitor loss (shown in W&B)")
    print(f"  - Test: Used ONLY after training for final evaluation")
    
    print(f"\n✓ Files created:")
    print(f"  - ../data/train.jsonl")
    print(f"  - ../data/validation.jsonl (for training monitoring)")
    print(f"  - ../data/test.jsonl (for final evaluation)")

if __name__ == "__main__":
    main()
