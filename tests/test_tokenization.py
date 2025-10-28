import pytest

# Skip entire module if transformers is not installed
transformers = pytest.importorskip("transformers")

from transformers import AutoTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    """Load a small tokenizer for testing."""
    # Use a small, fast tokenizer for testing
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def test_tokenization_short_text_not_excessively_padded(tokenizer):
    """Ensure short prompts aren't excessively padded when not needed."""
    short_text = "Test prompt."
    
    # Tokenize without padding
    tokens_no_pad = tokenizer(short_text, padding=False)
    
    # Tokenize with padding to max_length
    tokens_with_pad = tokenizer(short_text, padding="max_length", max_length=512)
    
    # The padded version should be longer
    assert len(tokens_with_pad['input_ids']) > len(tokens_no_pad['input_ids'])
    
    # The non-padded version should be reasonably short
    assert len(tokens_no_pad['input_ids']) < 20, "Short text produced too many tokens"


def test_tokenization_long_text_truncated_correctly(tokenizer):
    """Ensure long prompts are truncated correctly."""
    # Create a very long text
    long_text = "Medical entity extraction test. " * 200  # Repeat to make it long
    
    max_length = 128
    tokens = tokenizer(
        long_text,
        truncation=True,
        max_length=max_length,
        padding=False
    )
    
    # Should be truncated to max_length
    assert len(tokens['input_ids']) <= max_length
    assert len(tokens['input_ids']) == max_length  # Should be exactly max_length after truncation


def test_tokenization_preserves_special_medical_terms(tokenizer):
    """Test that medical terms are tokenized consistently."""
    medical_text_1 = "The patient has hypertension."
    medical_text_2 = "Diagnosed with hypertension."
    
    tokens_1 = tokenizer(medical_text_1, return_tensors=None)
    tokens_2 = tokenizer(medical_text_2, return_tensors=None)
    
    # Decode back to verify no information loss
    decoded_1 = tokenizer.decode(tokens_1['input_ids'], skip_special_tokens=True)
    decoded_2 = tokenizer.decode(tokens_2['input_ids'], skip_special_tokens=True)
    
    # The word "hypertension" should appear in both decoded texts
    assert "hypertension" in decoded_1.lower()
    assert "hypertension" in decoded_2.lower()


def test_tokenization_batch_processing(tokenizer):
    """Test that batch tokenization works correctly."""
    batch_texts = [
        "Patient has diabetes.",
        "Diagnosed with hypertension.",
        "Prescribed metformin for glucose control."
    ]
    
    # Tokenize as a batch
    batch_tokens = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # Should have same number of samples
    assert len(batch_tokens['input_ids']) == len(batch_texts)
    
    # All samples in batch should have same length (due to padding)
    lengths = [len(ids) for ids in batch_tokens['input_ids']]
    assert len(set(lengths)) == 1, "Batch samples have different lengths"


def test_tokenization_roundtrip_preserves_content(tokenizer):
    """Test that tokenize -> decode preserves the original content."""
    original_text = "Extract chemicals: aspirin, ibuprofen, metformin."
    
    tokens = tokenizer(original_text)
    decoded_text = tokenizer.decode(tokens['input_ids'], skip_special_tokens=True)
    
    # Core content should be preserved (may have minor whitespace differences)
    assert "aspirin" in decoded_text
    assert "ibuprofen" in decoded_text
    assert "metformin" in decoded_text
