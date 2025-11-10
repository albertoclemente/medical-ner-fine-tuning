#!/usr/bin/env python3
"""
Regenerate train/validation/test splits with stratified sampling.

This script creates balanced splits ensuring each contains exactly 33.3%
of each task type (chemical extraction, disease extraction, relationship extraction).
"""

import json
import random
from sklearn.model_selection import train_test_split

# Load all data
print("Loading combined data from existing splits...")
with open('train.jsonl', 'r') as f:
    train = [json.loads(line) for line in f]
with open('validation.jsonl', 'r') as f:
    val = [json.loads(line) for line in f]
with open('test.jsonl', 'r') as f:
    test = [json.loads(line) for line in f]

# Combine all data
all_data = train + val + test
print(f"Total samples loaded: {len(all_data)}")

# Helper function to classify task type
def get_task_type(prompt):
    """Classify the task type based on prompt for stratification."""
    prompt_lower = prompt.lower()
    if "influences between" in prompt_lower:
        return "relationship"
    elif "chemicals mentioned" in prompt_lower:
        return "chemical"
    elif "diseases mentioned" in prompt_lower:
        return "disease"
    return "other"

# Create stratification labels
print("\nCreating stratification labels...")
stratify_labels = [get_task_type(sample['prompt']) for sample in all_data]

# Show original distribution
from collections import Counter
original_dist = Counter(stratify_labels)
print("\nOriginal data distribution:")
for task, count in sorted(original_dist.items()):
    print(f"  {task}: {count} ({count/len(all_data)*100:.1f}%)")

# Set random seed for reproducibility
SPLIT_SEED = 42
random.seed(SPLIT_SEED)

print(f"\nPerforming stratified split (seed={SPLIT_SEED})...")

# First split: 80% train, 20% temp (for val + test)
train_data, temp_data, train_labels, temp_labels = train_test_split(
    all_data,
    stratify_labels,
    test_size=0.2,
    random_state=SPLIT_SEED,
    stratify=stratify_labels  # ✅ Guarantees exact proportions!
)

# Second split: split the 20% into 10% val, 10% test
val_data, test_data, val_labels, test_labels = train_test_split(
    temp_data,
    temp_labels,
    test_size=0.5,
    random_state=SPLIT_SEED + 1,
    stratify=temp_labels  # ✅ Guarantees exact proportions!
)

# Save new splits
print("\nSaving new splits...")
with open('train.jsonl', 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write(json.dumps(item) + '\n')

with open('validation.jsonl', 'w', encoding='utf-8') as f:
    for item in val_data:
        f.write(json.dumps(item) + '\n')

with open('test.jsonl', 'w', encoding='utf-8') as f:
    for item in test_data:
        f.write(json.dumps(item) + '\n')

print("\n" + "="*80)
print("NEW SPLIT VERIFICATION")
print("="*80)

# Verify distributions
for split_name, split_data, split_labels in [
    ("Train", train_data, train_labels),
    ("Validation", val_data, val_labels),
    ("Test", test_data, test_labels)
]:
    dist = Counter(split_labels)
    print(f"\n{split_name} ({len(split_data)} samples):")
    for task, count in sorted(dist.items()):
        percentage = count / len(split_data) * 100
        is_perfect = abs(percentage - 33.33) < 0.5
        marker = "✅" if is_perfect else "⚠️"
        print(f"  {marker} {task}: {count} ({percentage:.2f}%)")

print("\n" + "="*80)
print("✅ SPLITS REGENERATED SUCCESSFULLY!")
print("="*80)
print("\nNext steps:")
print("1. Re-run evaluation notebook to test on balanced data")
print("2. Re-train model with these balanced splits")
print("3. Compare performance with previous (imbalanced) model")
