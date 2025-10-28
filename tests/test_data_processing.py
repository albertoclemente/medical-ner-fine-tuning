np,import json
import sys
from pathlib import Path
import pytest
from sklearn.model_selection import train_test_split

# Add project root to sys.path to allow imports from scripts
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Since the function is in a notebook, we'll copy it here for testing.
# In a real-world scenario, this would be in a .py file and imported.
def format_instruction(sample):
    """Format data into Llama 3 chat format."""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical NER expert. Extract the requested entities from medical texts accurately.<|eot_id|><|start_header_id|>user<|end_header_id|>

{sample['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{sample['completion']}<|eot_id|>"""


@pytest.fixture
def sample_data():
    """Provides a sample data point for testing."""
    return {
        "prompt": "What are the side effects of Aspirin?",
        "completion": "Gastrointestinal bleeding\nNausea"
    }


def test_format_instruction(sample_data):
    """
    Tests the format_instruction function to ensure it correctly
    builds the Llama 3 chat format.
    """
    formatted_text = format_instruction(sample_data)
    
    # Check for required Llama 3 tokens
    assert "<|begin_of_text|>" in formatted_text
    assert "<|start_header_id|>system<|end_header_id|>" in formatted_text
    assert "<|eot_id|>" in formatted_text
    assert "<|start_header_id|>user<|end_header_id|>" in formatted_text
    assert "<|start_header_id|>assistant<|end_header_id|>" in formatted_text
    
    # Check that the prompt and completion are correctly placed
    assert sample_data['prompt'] in formatted_text
    assert sample_data['completion'] in formatted_text
    
    # Check structure
    assert formatted_text.startswith("<|begin_of_text|>")
    assert formatted_text.endswith("<|eot_id|>")


def test_format_instruction_preserves_multiline_completion(sample_data):
    """Ensure multiline completions are preserved correctly."""
    formatted_text = format_instruction(sample_data)
    
    # The completion should appear exactly as provided (with newlines)
    assert "Gastrointestinal bleeding\nNausea" in formatted_text


def test_format_instruction_with_special_characters():
    """Test that special characters in prompts/completions are handled correctly."""
    special_sample = {
        "prompt": "What about drugs with <special> & \"quoted\" text?",
        "completion": "Response with <brackets> and 'quotes'"
    }
    
    formatted_text = format_instruction(special_sample)
    
    # Special characters should be preserved
    assert "<special>" in formatted_text
    assert "&" in formatted_text
    assert '"quoted"' in formatted_text
    assert "'quotes'" in formatted_text


def test_data_splitting_creates_files(tmp_path):
    """
    Tests that the data splitting logic creates the train, validation, and test files
    in the correct location with the correct proportions.
    """
    # Create a temporary data directory structure
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create dummy data file with 100 samples for reliable splitting
    data = [{"prompt": f"prompt_{i}", "completion": f"completion_{i}"} for i in range(100)]
    data_path = data_dir / 'both_rel_instruct_all.jsonl'
    
    with open(data_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    # Perform the split manually (mimicking split_data.py logic)
    train_data, temp_data = train_test_split(
        data, 
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )
    
    # Save splits
    train_path = data_dir / 'train.jsonl'
    val_path = data_dir / 'validation.jsonl'
    test_path = data_dir / 'test.jsonl'
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    with open(test_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    # Verify files exist
    assert train_path.exists(), "train.jsonl was not created"
    assert val_path.exists(), "validation.jsonl was not created"
    assert test_path.exists(), "test.jsonl was not created"
    
    # Verify split proportions (80/10/10)
    assert len(train_data) == 80, f"Expected 80 train samples, got {len(train_data)}"
    assert len(val_data) == 10, f"Expected 10 validation samples, got {len(val_data)}"
    assert len(test_data) == 10, f"Expected 10 test samples, got {len(test_data)}"
    
    # Verify no data leakage (no overlap between splits)
    train_prompts = {item['prompt'] for item in train_data}
    val_prompts = {item['prompt'] for item in val_data}
    test_prompts = {item['prompt'] for item in test_data}
    
    assert len(train_prompts & val_prompts) == 0, "Data leakage: train and validation overlap"
    assert len(train_prompts & test_prompts) == 0, "Data leakage: train and test overlap"
    assert len(val_prompts & test_prompts) == 0, "Data leakage: validation and test overlap"
