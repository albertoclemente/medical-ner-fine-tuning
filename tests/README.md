# Test Suite for Medical NER Fine-Tuning Project

This directory contains comprehensive tests for the medical NER fine-tuning project.

## Test Categories

### 1. **Unit Tests** (Fast, No External Dependencies)
- `test_data_processing.py`: Tests data formatting and splitting logic
- `test_evaluation_metrics.py`: Tests precision, recall, and F1 score calculations

### 2. **Integration Tests** (Require Dependencies)
- `test_model_integration.py`: Tests the full training pipeline with a tiny model
- `test_tokenization.py`: Tests tokenization behavior
- `test_model_output.py`: Tests model output formatting and extraction
- `test_data_integrity.py`: Validates dataset file integrity

## Installation

Install test dependencies:

```bash
pip install -r requirements-test.txt
```

## Running Tests

### Run All Tests
```bash
pytest -v
```

### Run Only Fast Tests (Skip Slow Integration Tests)
```bash
pytest -v -m "not slow"
```

### Run Only Unit Tests
```bash
pytest -v tests/test_data_processing.py tests/test_evaluation_metrics.py
```

### Run with Coverage Report
```bash
pytest --cov=. --cov-report=html
```

### Run Specific Test File
```bash
pytest -v tests/test_evaluation_metrics.py
```

### Run Specific Test Function
```bash
pytest -v tests/test_evaluation_metrics.py::test_evaluation_metrics
```

## Test Markers

Tests are marked with custom markers to help you run specific subsets:

- `@pytest.mark.slow`: Tests that take longer to run (e.g., model loading)
- `@pytest.mark.integration`: Integration tests requiring external dependencies
- `@pytest.mark.requires_credentials`: Tests requiring HuggingFace or W&B credentials
- `@pytest.mark.requires_gpu`: Tests requiring GPU access

### Examples:

```bash
# Skip slow tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Skip tests requiring credentials
pytest -m "not requires_credentials"
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Pytest configuration and markers
├── README.md                      # This file
├── test_data_processing.py        # Data formatting and splitting tests
├── test_evaluation_metrics.py     # Metrics calculation tests
├── test_model_integration.py      # Full pipeline integration test
├── test_tokenization.py           # Tokenization logic tests
├── test_model_output.py           # Model output format tests
└── test_data_integrity.py         # Dataset validation tests
```

## Continuous Integration

For CI/CD pipelines, use:

```bash
# Fast tests only (for quick feedback)
pytest -v -m "not slow and not requires_credentials"

# Full test suite (for comprehensive validation)
pytest -v --cov=. --cov-report=xml
```

## Troubleshooting

### Import Errors
If you see import errors for `transformers`, `peft`, etc., ensure you've installed test dependencies:
```bash
pip install -r requirements-test.txt
```

### GPU Tests Failing
Some tests may require GPU access. Skip them with:
```bash
pytest -m "not requires_gpu"
```

### Data File Not Found
Some tests require the `both_rel_instruct_all.jsonl` file in the `data/` directory. If the file is missing, those tests will be skipped automatically.

## Writing New Tests

When adding new tests:

1. Use descriptive test names: `test_<what_is_being_tested>`
2. Add appropriate markers (`@pytest.mark.slow`, etc.)
3. Use fixtures from `conftest.py` for shared setup
4. Document what the test validates in the docstring
5. Use `pytest.skip()` if dependencies are missing

Example:
```python
import pytest

@pytest.mark.slow
@pytest.mark.integration
def test_new_feature():
    """Test that new feature works correctly."""
    # Your test code here
    assert True
```
