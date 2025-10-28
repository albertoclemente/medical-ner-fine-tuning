import pytest

# This logic is copied from the Medical_NER_Evaluation.ipynb notebook.
# In a production setting, this would be in a shared utility script.
def calculate_metrics(expected_items, predicted_items):
    """Calculates precision, recall, and F1 score for a single sample."""
    common = expected_items & predicted_items
    
    precision = len(common) / len(predicted_items) if len(predicted_items) > 0 else 0
    recall = len(common) / len(expected_items) if len(expected_items) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": len(common),
        "predicted": len(predicted_items),
        "expected": len(expected_items)
    }

@pytest.mark.parametrize("expected, predicted, expected_metrics", [
    # Case 1: Perfect match
    (
        {'aspirin', 'ibuprofen'}, 
        {'aspirin', 'ibuprofen'}, 
        {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    ),
    # Case 2: Model missed one (false negative)
    (
        {'aspirin', 'ibuprofen', 'metformin'},
        {'aspirin', 'ibuprofen'},
        {'precision': 1.0, 'recall': 2/3, 'f1': 0.8}
    ),
    # Case 3: Model predicted an extra one (false positive)
    (
        {'aspirin', 'ibuprofen'},
        {'aspirin', 'ibuprofen', 'paracetamol'},
        {'precision': 2/3, 'recall': 1.0, 'f1': 0.8}
    ),
    # Case 4: Partial overlap
    (
        {'aspirin', 'metformin'},
        {'aspirin', 'paracetamol'},
        {'precision': 0.5, 'recall': 0.5, 'f1': 0.5}
    ),
    # Case 5: No overlap
    (
        {'aspirin', 'ibuprofen'},
        {'paracetamol', 'metformin'},
        {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    ),
    # Case 6: Empty prediction
    (
        {'aspirin'},
        set(),
        {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    ),
    # Case 7: Empty ground truth (should not happen, but good to test)
    (
        set(),
        {'aspirin'},
        {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    )
])
def test_evaluation_metrics(expected, predicted, expected_metrics):
    """
    Tests the metrics calculation logic with various scenarios.
    """
    metrics = calculate_metrics(set(expected), set(predicted))
    
    assert pytest.approx(metrics['precision']) == expected_metrics['precision']
    assert pytest.approx(metrics['recall']) == expected_metrics['recall']
    assert pytest.approx(metrics['f1']) == expected_metrics['f1']

