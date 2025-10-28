import json
import pytest
from pathlib import Path
from collections import Counter

# Get the project root
project_root = Path(__file__).parent.parent


@pytest.fixture
def data_dir():
    """Return the path to the data directory."""
    return project_root / "data"


@pytest.fixture
def source_data_path(data_dir):
    """Return the path to the source data file."""
    return data_dir / "both_rel_instruct_all.jsonl"


def test_source_data_file_exists(source_data_path):
    """Verify the source data file exists."""
    assert source_data_path.exists(), f"Source data file not found: {source_data_path}"


def test_source_data_all_lines_valid_json(source_data_path):
    """Check that all lines in the source data are valid JSON."""
    if not source_data_path.exists():
        pytest.skip(f"Source data file not found: {source_data_path}")
    
    invalid_lines = []
    
    with open(source_data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                invalid_lines.append((line_num, str(e)))
    
    assert len(invalid_lines) == 0, f"Found {len(invalid_lines)} invalid JSON lines: {invalid_lines[:5]}"


def test_source_data_required_keys(source_data_path):
    """Check that all samples have required 'prompt' and 'completion' keys."""
    if not source_data_path.exists():
        pytest.skip(f"Source data file not found: {source_data_path}")
    
    missing_keys = []
    
    with open(source_data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line)
                if 'prompt' not in sample:
                    missing_keys.append((line_num, 'prompt'))
                if 'completion' not in sample:
                    missing_keys.append((line_num, 'completion'))
            except json.JSONDecodeError:
                # This is caught by another test
                pass
    
    assert len(missing_keys) == 0, f"Found {len(missing_keys)} samples with missing keys: {missing_keys[:10]}"


def test_source_data_no_empty_values(source_data_path):
    """Check that prompts and completions are not empty strings."""
    if not source_data_path.exists():
        pytest.skip(f"Source data file not found: {source_data_path}")
    
    empty_values = []
    
    with open(source_data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line)
                if not sample.get('prompt', '').strip():
                    empty_values.append((line_num, 'prompt'))
                if not sample.get('completion', '').strip():
                    empty_values.append((line_num, 'completion'))
            except json.JSONDecodeError:
                pass
    
    assert len(empty_values) == 0, f"Found {len(empty_values)} samples with empty values: {empty_values[:10]}"


def test_source_data_check_for_duplicates(source_data_path):
    """Check for duplicate samples (same prompt)."""
    if not source_data_path.exists():
        pytest.skip(f"Source data file not found: {source_data_path}")
    
    prompts = []
    
    with open(source_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line)
                prompts.append(sample.get('prompt', ''))
            except json.JSONDecodeError:
                pass
    
    # Count duplicates
    prompt_counts = Counter(prompts)
    duplicates = {prompt: count for prompt, count in prompt_counts.items() if count > 1}
    
    # We allow some duplicates in medical data (same prompt, different task)
    # but warn if there are too many
    if duplicates:
        total_duplicates = sum(count - 1 for count in duplicates.values())
        duplicate_percentage = (total_duplicates / len(prompts)) * 100
        
        # Warn if more than 10% are duplicates
        assert duplicate_percentage < 10, (
            f"High duplicate rate: {duplicate_percentage:.1f}% "
            f"({total_duplicates} duplicates out of {len(prompts)} samples)"
        )


def test_source_data_task_distribution(source_data_path):
    """Verify that the dataset contains different types of tasks."""
    if not source_data_path.exists():
        pytest.skip(f"Source data file not found: {source_data_path}")
    
    task_keywords = {
        'chemical': 0,
        'disease': 0,
        'relationship': 0,
        'influence': 0,
    }
    total_samples = 0
    
    with open(source_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line)
                prompt = sample.get('prompt', '').lower()
                total_samples += 1
                
                if 'chemical' in prompt:
                    task_keywords['chemical'] += 1
                if 'disease' in prompt:
                    task_keywords['disease'] += 1
                if 'relationship' in prompt or 'influence' in prompt:
                    task_keywords['relationship'] += 1
                    
            except json.JSONDecodeError:
                pass
    
    # Ensure we have a variety of tasks (at least 2 types)
    tasks_present = sum(1 for count in task_keywords.values() if count > 0)
    assert tasks_present >= 2, (
        f"Dataset should contain at least 2 task types, found {tasks_present}. "
        f"Distribution: {task_keywords}"
    )


def test_split_files_integrity(data_dir):
    """Verify that train/val/test split files are well-formed if they exist."""
    split_files = ['train.jsonl', 'validation.jsonl', 'test.jsonl']
    
    for split_file in split_files:
        file_path = data_dir / split_file
        
        if not file_path.exists():
            continue  # Skip if file doesn't exist yet
        
        # Check all lines are valid JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line)
                    assert 'prompt' in sample, f"{split_file} line {line_num}: missing 'prompt'"
                    assert 'completion' in sample, f"{split_file} line {line_num}: missing 'completion'"
                except json.JSONDecodeError as e:
                    pytest.fail(f"{split_file} line {line_num}: invalid JSON - {e}")
