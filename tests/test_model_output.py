import pytest
from unittest.mock import Mock, patch

# Skip entire module if transformers is not installed
transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from transformers import AutoTokenizer


def generate_response(model, tokenizer, prompt_text, max_new_tokens=512):
    """
    Generate a response for a given prompt.
    This is the function from the evaluation notebook.
    """
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical NER expert. Extract the requested entities from medical texts accurately.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant's response
    if "assistant\n\n" in response:
        response = response.split("assistant\n\n")[-1]
    elif "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    return response.strip()


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.eos_token_id = 0
    tokenizer.pad_token_id = 0
    
    # Mock the tokenizer call
    def tokenize_side_effect(text, return_tensors=None):
        mock_output = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        if return_tensors == "pt":
            mock_output['input_ids'] = mock_output['input_ids']
            mock_output['attention_mask'] = mock_output['attention_mask']
        
        # Add .to() method to the dict
        class TensorDict(dict):
            def to(self, device):
                return self
        
        return TensorDict(mock_output)
    
    tokenizer.side_effect = tokenize_side_effect
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.device = "cpu"
    
    # Mock generate method to return a fixed output
    def generate_side_effect(**kwargs):
        # Return a tensor that represents a tokenized response
        return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    
    model.generate = Mock(side_effect=generate_side_effect)
    return model


def test_generate_response_formats_prompt_correctly(mock_model, mock_tokenizer):
    """Test that the prompt is correctly formatted with Llama 3 tags."""
    prompt = "What chemicals are mentioned in this text?"
    
    # We'll capture what was passed to the tokenizer
    tokenizer_calls = []
    
    def capture_tokenizer_call(text, return_tensors=None):
        tokenizer_calls.append(text)
        mock_output = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        
        class TensorDict(dict):
            def to(self, device):
                return self
        
        return TensorDict(mock_output)
    
    mock_tokenizer.side_effect = capture_tokenizer_call
    
    # Mock the decode to return a simple response
    mock_tokenizer.decode = Mock(return_value="assistant\n\naspirin\nibuprofen")
    
    response = generate_response(mock_model, mock_tokenizer, prompt)
    
    # Check that the tokenizer was called
    assert len(tokenizer_calls) > 0
    formatted_prompt = tokenizer_calls[0]
    
    # Verify Llama 3 format
    assert "<|begin_of_text|>" in formatted_prompt
    assert "<|start_header_id|>system<|end_header_id|>" in formatted_prompt
    assert "<|start_header_id|>user<|end_header_id|>" in formatted_prompt
    assert "<|start_header_id|>assistant<|end_header_id|>" in formatted_prompt
    assert "<|eot_id|>" in formatted_prompt
    assert prompt in formatted_prompt


def test_generate_response_extracts_assistant_reply(mock_model, mock_tokenizer):
    """Test that the function correctly extracts the assistant's response."""
    # Mock tokenizer to return a full conversation
    full_response = """<|begin_of_text|>system

You are a medical NER expert.

user

What are the chemicals?

assistant

aspirin
ibuprofen
metformin"""
    
    mock_tokenizer.decode = Mock(return_value=full_response)
    
    response = generate_response(mock_model, mock_tokenizer, "What are the chemicals?")
    
    # Should extract only the assistant's part
    assert "aspirin" in response
    assert "ibuprofen" in response
    assert "metformin" in response
    assert "system" not in response.lower()
    assert "user" not in response.lower()


def test_generate_response_handles_assistant_with_double_newline(mock_model, mock_tokenizer):
    """Test extraction when 'assistant\\n\\n' pattern is present."""
    full_response = "Some preamble text assistant\n\nchemical1\nchemical2"
    
    mock_tokenizer.decode = Mock(return_value=full_response)
    
    response = generate_response(mock_model, mock_tokenizer, "Extract chemicals")
    
    # Should extract text after "assistant\n\n"
    assert response == "chemical1\nchemical2"
    assert "preamble" not in response


def test_generate_response_strips_whitespace(mock_model, mock_tokenizer):
    """Test that leading/trailing whitespace is removed."""
    full_response = "   assistant\n\n   aspirin   \n   ibuprofen   "
    
    mock_tokenizer.decode = Mock(return_value=full_response)
    
    response = generate_response(mock_model, mock_tokenizer, "List chemicals")
    
    # Should be stripped
    assert not response.startswith(" ")
    assert not response.endswith(" ")


def test_generate_response_preserves_multiline_structure(mock_model, mock_tokenizer):
    """Test that newline-separated entities are preserved."""
    full_response = "assistant\n\nchemical1\nchemical2\nchemical3"
    
    mock_tokenizer.decode = Mock(return_value=full_response)
    
    response = generate_response(mock_model, mock_tokenizer, "Extract chemicals")
    
    # Should preserve newlines
    lines = response.split('\n')
    assert len(lines) >= 3
    assert "chemical1" in response
    assert "chemical2" in response
    assert "chemical3" in response


@pytest.mark.integration
def test_generate_response_with_real_tokenizer():
    """Integration test with a real tokenizer (but still mocked model)."""
    # Use a real tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Mock model
    mock_model = Mock()
    mock_model.device = "cpu"
    
    # Mock generate to return token IDs that decode to our expected response
    expected_response = "aspirin\nibuprofen\nmetformin"
    response_tokens = tokenizer.encode(f"assistant\n\n{expected_response}")
    
    mock_model.generate = Mock(return_value=torch.tensor([response_tokens]))
    
    response = generate_response(mock_model, tokenizer, "What are the chemicals?")
    
    # Check that we got a reasonable response
    assert isinstance(response, str)
    assert len(response) > 0
