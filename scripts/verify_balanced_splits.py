#!/usr/bin/env python3
"""
Verify that train/validation/test splits have balanced task distribution.

This script checks if the data splits have properly balanced distributions of:
- Chemical extraction tasks
- Disease extraction tasks  
- Relationship extraction tasks

Run this AFTER creating new splits with shuffle=True to verify the fix worked.
"""

import json
import sys
from pathlib import Path

def get_task_type(prompt):
    """Identify the task type from the prompt text."""
    prompt_lower = prompt.lower()
    if "influences between" in prompt_lower:
        return "relationship"
    elif "chemicals mentioned" in prompt_lower:
        return "chemical"
    elif "diseases mentioned" in prompt_lower:
        return "disease"
    return "other"

def analyze_split(filepath):
    """Analyze task distribution in a data split."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    tasks = {}
    for sample in data:
        task = get_task_type(sample['prompt'])
        tasks[task] = tasks.get(task, 0) + 1
    
    return data, tasks

def main():
    # Find the notebooks directory
    script_dir = Path(__file__).parent
    notebooks_dir = script_dir.parent / 'notebooks'
    
    print("="*80)
    print("DATA SPLIT BALANCE VERIFICATION")
    print("="*80)
    
    all_balanced = True
    
    for filename in ['train.jsonl', 'validation.jsonl', 'test.jsonl']:
        filepath = notebooks_dir / filename
        
        if not filepath.exists():
            print(f"\n❌ {filename} not found at {filepath}")
            continue
        
        data, tasks = analyze_split(filepath)
        
        print(f"\n{filename} ({len(data)} samples):")
        for task, count in sorted(tasks.items()):
            percentage = count / len(data) * 100
            print(f"  {task}: {count} ({percentage:.1f}%)")
            
            # Check for severe imbalance (>80% or <10% for any task)
            if percentage > 80:
                print(f"    ⚠️  WARNING: {task} is over-represented ({percentage:.1f}%)")
                all_balanced = False
            elif percentage < 10 and task != 'other':
                print(f"    ⚠️  WARNING: {task} is under-represented ({percentage:.1f}%)")
                all_balanced = False
    
    print("\n" + "="*80)
    if all_balanced:
        print("✅ VERIFICATION PASSED")
        print("All splits have balanced task distributions!")
        print("\nExpected distributions (~33% each task):")
        print("  ✓ No task is over-represented (>80%)")
        print("  ✓ No task is under-represented (<10%)")
        print("\n🎉 Data is ready for training!")
        return 0
    else:
        print("❌ VERIFICATION FAILED")
        print("Data splits are NOT balanced!")
        print("\nPossible issues:")
        print("  - shuffle=False in train_test_split()")
        print("  - Data not randomized before splitting")
        print("\n💡 Fix: Use shuffle=True in Medical_NER_Fine_Tuning_RUN.ipynb")
        return 1

if __name__ == "__main__":
    sys.exit(main())
