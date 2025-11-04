"""
Split the medical NER dataset into train, validation, and test sets using STRATIFIED sampling.
- Train: 80% (2,400 samples) - for fine-tuning
- Validation: 10% (300 samples) - for monitoring during training (W&B)
- Test: 10% (300 samples) - for final evaluation after training

IMPORTANT: Uses stratification to ensure balanced task distribution across all splits!
"""

import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

def get_task_type(sample):
    """Classify the task type based on prompt for stratification."""
    prompt = sample.get('prompt', '')
    if 'chemicals mentioned' in prompt:
        return 'chemical'
    elif 'diseases mentioned' in prompt:
        return 'disease'
    else:
        return 'other'

def main():
    print("="*70)
    print("STRATIFIED DATA SPLITTING FOR MEDICAL NER")
    print("="*70)
    
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"

    # Load data
    print("\nLoading dataset...")
    with open(data_dir / 'both_rel_instruct_all.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Total samples loaded: {len(data)}")
    
    # Create stratification labels
    labels = [get_task_type(sample) for sample in data]
    label_dist = Counter(labels)
    
    print(f"\nOriginal task distribution:")
    for task, count in sorted(label_dist.items()):
        print(f"  {task}: {count} ({count/len(data)*100:.1f}%)")
    
    # First split: 80% train, 20% temp (for val + test)
    # Using stratify ensures EXACT proportions in both splits
    print(f"\nPerforming stratified split...")
    random.seed(42)
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data,
        labels,
        test_size=0.2,  # 20% for val + test
        random_state=42,
        stratify=labels  # ✅ GUARANTEES proportional distribution!
    )
    
    # Second split: split the 20% into 10% val, 10% test
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data,
        temp_labels,
        test_size=0.5,  # 50% of 20% = 10% of total
        random_state=43,
        stratify=temp_labels  # ✅ GUARANTEES proportional distribution!
    )
    
    # Save train set
    with open(data_dir / 'train.jsonl', 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    # Save validation set
    with open(data_dir / 'validation.jsonl', 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    # Save test set
    with open(data_dir / 'test.jsonl', 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    # Verify splits
    print(f"\n{'='*70}")
    print("SPLIT VERIFICATION")
    print(f"{'='*70}")
    
    for split_name, split_data, split_labels in [
        ("Train", train_data, train_labels),
        ("Validation", val_data, val_labels),
        ("Test", test_data, test_labels)
    ]:
        label_dist = Counter(split_labels)
        print(f"\n{split_name} ({len(split_data)} samples):")
        for task, count in sorted(label_dist.items()):
            percentage = count / len(split_data) * 100
            expected = label_dist[task] / len(data) * 100
            print(f"  {task:10s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\n{'='*70}")
    print("✓ Successfully split dataset with stratification!")
    print(f"{'='*70}")
    print(f"  - Train samples: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  - Validation samples: {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")
    print(f"  - Test samples: {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")
    
    print(f"\n✓ Files created:")
    print(f"  - {data_dir / 'train.jsonl'}")
    print(f"  - {data_dir / 'validation.jsonl'}")
    print(f"  - {data_dir / 'test.jsonl'}")
    
    print(f"\n✅ All splits now have proportional task distributions!")

if __name__ == "__main__":
    main()
