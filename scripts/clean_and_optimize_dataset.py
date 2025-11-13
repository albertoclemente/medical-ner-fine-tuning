#!/usr/bin/env python3
"""
Dataset Cleaning and Optimization Script
=========================================

Fixes all identified dataset quality issues:
1. Remove samples with empty completions
2. Remove samples with 0 entities
3. Validate and truncate long prompts
4. Normalize entity formatting
5. Create cleaned dataset splits

Usage:
    python scripts/clean_and_optimize_dataset.py
"""

import json
import re
from pathlib import Path
from collections import Counter

# Configuration
SOURCE_DIR = Path("data/splits_20251111")
OUTPUT_DIR = Path("data/splits_cleaned_20251113")
MAX_PROMPT_LENGTH = 2048  # Max tokens for Llama model (roughly 2048 chars)

# Statistics
stats = {
    'original_count': 0,
    'removed_empty': 0,
    'removed_zero_entities': 0,
    'truncated_prompts': 0,
    'normalized_entities': 0,
    'final_count': 0
}

def parse_entities(completion: str) -> list:
    """Extract entity lines from completion."""
    lines = completion.split('\n')
    entities = []
    for line in lines:
        line = line.strip()
        if line.startswith('-') or line.startswith('•') or line.startswith('*'):
            entity = line.lstrip('-•* ').strip()
            if entity:
                entities.append(entity)
    return entities

def normalize_entity(entity: str) -> str:
    """
    Normalize entity formatting:
    - Remove extra whitespace
    - Standardize special characters
    - Keep hyphenated forms
    """
    # Remove extra whitespace
    entity = ' '.join(entity.split())
    
    # Normalize quotes
    entity = entity.replace('"', '"').replace('"', '"')
    entity = entity.replace("'", "'").replace("'", "'")
    
    return entity

def truncate_prompt(prompt: str, max_length: int = MAX_PROMPT_LENGTH) -> tuple[str, bool]:
    """
    Truncate long prompts intelligently.
    Returns: (truncated_prompt, was_truncated)
    """
    if len(prompt) <= max_length:
        return prompt, False
    
    # Split into instruction and article
    parts = prompt.split('\n\n', 1)
    if len(parts) == 2:
        instruction = parts[0]
        article = parts[1]
        
        # Calculate remaining space for article
        remaining = max_length - len(instruction) - 4  # 4 for \n\n
        
        if remaining > 100:  # Keep at least 100 chars of article
            # Truncate article at sentence boundary
            truncated_article = article[:remaining]
            last_period = truncated_article.rfind('.')
            if last_period > remaining * 0.7:  # If we can keep 70% with sentence boundary
                truncated_article = truncated_article[:last_period + 1]
            
            return f"{instruction}\n\n{truncated_article}", True
    
    # Fallback: simple truncation
    return prompt[:max_length], True

def validate_sample(sample: dict) -> tuple[bool, str]:
    """
    Validate a single sample.
    Returns: (is_valid, reason_if_invalid)
    """
    # Check for required keys
    if 'prompt' not in sample or 'completion' not in sample:
        return False, "missing_keys"
    
    # Check for empty prompt
    if not sample['prompt'].strip():
        return False, "empty_prompt"
    
    # Check for empty completion
    if not sample['completion'].strip():
        return False, "empty_completion"
    
    # Check for zero entities
    entities = parse_entities(sample['completion'])
    if len(entities) == 0:
        return False, "zero_entities"
    
    return True, ""

def clean_sample(sample: dict) -> dict:
    """Clean and normalize a single sample."""
    cleaned = sample.copy()
    
    # Truncate long prompts
    cleaned['prompt'], was_truncated = truncate_prompt(sample['prompt'])
    if was_truncated:
        stats['truncated_prompts'] += 1
    
    # Normalize entities in completion
    entities = parse_entities(sample['completion'])
    normalized_entities = [normalize_entity(e) for e in entities]
    
    # Check if any normalization occurred
    if normalized_entities != entities:
        stats['normalized_entities'] += 1
    
    # Rebuild completion with normalized entities
    cleaned['completion'] = '\n'.join(f"- {e}" for e in normalized_entities)
    
    return cleaned

def process_file(input_path: Path, output_path: Path):
    """Process a single JSONL file."""
    print(f"\nProcessing: {input_path.name}")
    print("-" * 80)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]
    
    local_stats = {
        'original': len(samples),
        'removed_empty': 0,
        'removed_zero': 0,
        'kept': 0
    }
    
    stats['original_count'] += len(samples)
    
    # Filter and clean
    cleaned_samples = []
    for sample in samples:
        is_valid, reason = validate_sample(sample)
        
        if not is_valid:
            if reason == "empty_completion":
                stats['removed_empty'] += 1
                local_stats['removed_empty'] += 1
            elif reason == "zero_entities":
                stats['removed_zero_entities'] += 1
                local_stats['removed_zero'] += 1
            continue
        
        # Clean the valid sample
        cleaned = clean_sample(sample)
        cleaned_samples.append(cleaned)
        local_stats['kept'] += 1
    
    stats['final_count'] += len(cleaned_samples)
    
    # Save cleaned data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in cleaned_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Print stats
    print(f"  Original samples: {local_stats['original']}")
    print(f"  Removed (empty): {local_stats['removed_empty']}")
    print(f"  Removed (0 entities): {local_stats['removed_zero']}")
    print(f"  Kept (cleaned): {local_stats['kept']}")
    print(f"  ✓ Saved to: {output_path}")

def analyze_cleaned_data(output_dir: Path):
    """Analyze the cleaned dataset."""
    print("\n" + "=" * 80)
    print("CLEANED DATASET ANALYSIS")
    print("=" * 80)
    
    for split in ['train', 'validation', 'test']:
        file_path = output_dir / f"{split}.jsonl"
        if not file_path.exists():
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f]
        
        # Count entities per sample
        entity_counts = []
        task_counts = Counter()
        
        for sample in samples:
            entities = parse_entities(sample['completion'])
            entity_counts.append(len(entities))
            
            # Determine task
            prompt_lower = sample['prompt'].lower()
            if 'influences between' in prompt_lower:
                task_counts['influences'] += 1
            elif 'chemicals mentioned' in prompt_lower:
                task_counts['chemicals'] += 1
            elif 'diseases mentioned' in prompt_lower:
                task_counts['diseases'] += 1
        
        print(f"\n{split.upper()}:")
        print(f"  Total samples: {len(samples)}")
        print(f"  Task distribution:")
        for task, count in sorted(task_counts.items()):
            print(f"    {task}: {count} ({count/len(samples)*100:.1f}%)")
        print(f"  Entities per sample:")
        print(f"    Min: {min(entity_counts)}, Max: {max(entity_counts)}, Avg: {sum(entity_counts)/len(entity_counts):.1f}")

def main():
    print("=" * 80)
    print("DATASET CLEANING AND OPTIMIZATION")
    print("=" * 80)
    print(f"\nSource: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Max prompt length: {MAX_PROMPT_LENGTH} chars")
    
    # Process each split
    for split in ['train', 'validation', 'test']:
        input_file = SOURCE_DIR / f"{split}.jsonl"
        output_file = OUTPUT_DIR / f"{split}.jsonl"
        
        if input_file.exists():
            process_file(input_file, output_file)
        else:
            print(f"\n⚠️  Warning: {input_file} not found, skipping...")
    
    # Print overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"\nTotal original samples: {stats['original_count']}")
    print(f"\nCleaning actions:")
    print(f"  • Removed (empty completions): {stats['removed_empty']}")
    print(f"  • Removed (zero entities): {stats['removed_zero_entities']}")
    print(f"  • Truncated (long prompts): {stats['truncated_prompts']}")
    print(f"  • Normalized (entities): {stats['normalized_entities']}")
    print(f"\nFinal clean samples: {stats['final_count']}")
    print(f"Data retention rate: {stats['final_count']/stats['original_count']*100:.1f}%")
    
    # Analyze cleaned data
    analyze_cleaned_data(OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("✓ Dataset cleaning complete!")
    print("=" * 80)
    print(f"\n📁 Cleaned data saved to: {OUTPUT_DIR}")
    print("\n💡 Next steps:")
    print("   1. Review the cleaned data in data/splits_cleaned_20251113/")
    print("   2. Update training script to use cleaned data")
    print("   3. Retrain model with cleaned dataset")
    print("=" * 80)

if __name__ == "__main__":
    main()
