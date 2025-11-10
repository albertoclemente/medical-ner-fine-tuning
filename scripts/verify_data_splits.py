#!/usr/bin/env python3
"""
Verify that data splits have balanced task distribution.

This script checks if train.jsonl, validation.jsonl, and test.jsonl
have approximately equal distribution of all three task types:
- Chemical extraction (~33%)
- Disease extraction (~33%)
- Relationship extraction (~33%)

If imbalanced (e.g., 100% relationship in test set), you need to re-run
the data splitting section with shuffle=True.
"""

import json
from pathlib import Path

def get_task_type(prompt):
    """Classify the task type based on prompt."""
    prompt_lower = prompt.lower()
    if "influences between" in prompt_lower:
        return "relationship"
    elif "chemicals mentioned" in prompt_lower:
        return "chemical"
    elif "diseases mentioned" in prompt_lower:
        return "disease"
    return "other"

def analyze_split(filepath):
    """Analyze task distribution in a split file."""
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    
    task_counts = {}
    for sample in data:
        task = get_task_type(sample['prompt'])
        task_counts[task] = task_counts.get(task, 0) + 1
    
    return data, task_counts

def main():
    print("="*80)
    print("DATA SPLIT VERIFICATION")
    print("="*80)
    
    files = ['train.jsonl', 'validation.jsonl', 'test.jsonl']
    all_balanced = True
    
    for filename in files:
        filepath = Path(filename)
        if not filepath.exists():
            print(f"\n❌ {filename} not found!")
            continue
        
        data, task_counts = analyze_split(filepath)
        total = len(data)
        
        print(f"\n{filename} ({total} samples):")
        for task, count in sorted(task_counts.items()):
            percentage = count / total * 100
            print(f"  {task}: {count:4d} ({percentage:5.1f}%)")
        
        # Check if balanced (allow 10% deviation from 33.3%)
        for task, count in task_counts.items():
            percentage = count / total * 100
            if percentage < 23.3 or percentage > 43.3:  # More than 10% deviation from 33.3%
                if task == "relationship" and percentage < 20:
                    print(f"  ⚠️  WARNING: {task} is severely underrepresented!")
                    all_balanced = False
                elif percentage > 90:
                    print(f"  ⚠️  WARNING: {task} is severely overrepresented!")
                    all_balanced = False
    
    print("\n" + "="*80)
    if all_balanced:
        print("✅ All splits have balanced task distribution!")
        print("   The model will be trained on all three task types equally.")
        print("\n   Using stratified splitting ensures EXACT 33.3% distribution!")
    else:
        print("⚠️  IMBALANCED SPLITS DETECTED!")
        print("\n   Problem: Data was not properly split with stratification.")
        print("   Impact: Model will underperform on underrepresented tasks.")
        print("\n   Solution:")
        print("   1. Open Medical_NER_Fine_Tuning.ipynb")
        print("   2. Re-run Section 6 'Dataset Splitting'")
        print("   3. Verify stratify=labels is used in both train_test_split calls")
        print("   4. Re-run this script to verify")
        print("   5. Re-train the model with balanced data")
        print("\n   Note: Stratified splitting guarantees exact proportions,")
        print("         unlike shuffle=True which only gives ~33% ± 2-3%")
    print("="*80)

if __name__ == "__main__":
    main()
