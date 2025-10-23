# Model Validation Strategy

## ✅ Proper Validation on Unseen Data

### The Problem with Training Data Validation
**DON'T DO THIS**: Testing on data the model has already seen during training
- ❌ Gives inflated performance metrics
- ❌ Doesn't measure true generalization ability
- ❌ Can't detect overfitting
- ❌ Not a true measure of real-world performance

### Our Solution: True Held-Out Validation
**DO THIS**: Test on completely unseen data from the validation set
- ✅ Data split: 90% train (2,700 samples) / 10% validation (300 samples)
- ✅ Validation data is **never** used during training
- ✅ Provides accurate measure of generalization
- ✅ Detects overfitting
- ✅ Realistic performance estimate

---

## Data Flow

```
both_rel_instruct_all.jsonl (3,000 samples)
           |
           v
    [split_data.py]
           |
           +---> train.jsonl (2,700 samples)
           |           |
           |           v
           |     [Used in training]
           |     Model learns from these
           |
           +---> validation.jsonl (300 samples)
                       |
                       v
                 [NEVER used in training]
                 Used ONLY for evaluation
```

---

## Validation Implementation

### In Jupyter Notebook (Section 16)

```python
# Load UNSEEN validation data
with open('validation.jsonl', 'r', encoding='utf-8') as f:
    validation_data = [json.loads(line) for line in f]

# Test on samples that were NOT seen during training
for sample in validation_data[:5]:
    prediction = generate_response(sample['prompt'])
    # Compare prediction with expected output
```

**Key Points**:
- ✅ Loads validation.jsonl (NOT train.jsonl)
- ✅ These 300 samples were held out during training
- ✅ Model has never seen these examples before
- ✅ True test of generalization ability

### In Standalone Script (validate_model.py)

```bash
python validate_model.py
```

**What it does**:
1. Loads the fine-tuned model
2. Loads **unseen** validation data from `validation.jsonl`
3. Tests on 5 random validation samples
4. Computes accuracy metrics
5. Shows custom test cases

---

## Evaluation Metrics

### Per-Sample Metrics
For each validation sample, we compute:

```python
expected_items = set(ground_truth.split('\n'))
predicted_items = set(model_output.split('\n'))

correct = expected_items & predicted_items
missed = expected_items - predicted_items
extra = predicted_items - expected_items

accuracy = len(correct) / len(expected_items) * 100
```

**Metrics Displayed**:
- ✓ **Correct extractions**: How many entities were correctly identified
- ✗ **Missed extractions**: Entities in ground truth but not predicted
- ⚠ **Extra extractions**: Entities predicted but not in ground truth
- 📈 **Accuracy**: Percentage of correct extractions

### Example Output

```
📊 EVALUATION METRICS:
  ✓ Correct extractions: 8/10
  ✗ Missed extractions: 2
  ⚠ Extra extractions: 1
  📈 Accuracy: 80.0%

  Missed items: ['chronic kidney disease', 'proteinuria']
  Extra items: ['renal insufficiency']
```

---

## Three Levels of Validation

### Level 1: During Training (Automatic)
- **When**: Every 100 steps during training
- **Data**: Validation set (300 samples)
- **Metric**: Validation loss
- **Purpose**: Monitor for overfitting in real-time

```python
# In TrainingArguments
eval_strategy="steps",
eval_steps=100,
```

### Level 2: Post-Training Validation (Section 16 / validate_model.py)
- **When**: After training completes
- **Data**: Same validation set (300 samples)
- **Metrics**: Accuracy, precision, recall per sample
- **Purpose**: Detailed performance analysis on held-out data

### Level 3: Custom Test Cases (Section 17)
- **When**: After validation
- **Data**: Completely new medical texts you write
- **Metrics**: Qualitative assessment
- **Purpose**: Test on novel scenarios and edge cases

---

## Why This Matters

### Example Scenario

**Bad Validation** (Testing on Training Data):
```python
# ❌ WRONG: Testing on data used in training
for sample in train_data[:5]:  # These were used in training!
    prediction = model(sample['prompt'])
```

**Result**: Model gets 95% accuracy (memorization, not learning)

---

**Good Validation** (Testing on Unseen Data):
```python
# ✅ CORRECT: Testing on held-out validation data
for sample in validation_data[:5]:  # Never seen during training!
    prediction = model(sample['prompt'])
```

**Result**: Model gets 80% accuracy (true generalization ability)

---

## Best Practices

### ✅ DO:
1. **Always split data before training**
   ```bash
   python split_data.py  # Creates train.jsonl and validation.jsonl
   ```

2. **Use validation.jsonl for evaluation**
   ```python
   validation_data = load_jsonl('validation.jsonl')
   ```

3. **Keep validation data separate**
   - Don't mix train and validation
   - Don't use validation for hyperparameter tuning without cross-validation

4. **Report honest metrics**
   - Use metrics from validation set, not training set
   - Acknowledge when model struggles

### ❌ DON'T:
1. **Don't test on training data**
   ```python
   # ❌ BAD
   for sample in train_data:
       evaluate(sample)
   ```

2. **Don't use validation data for training**
   ```python
   # ❌ BAD
   all_data = train_data + validation_data
   train(all_data)
   ```

3. **Don't overfit to validation set**
   - Don't repeatedly adjust model based on validation performance
   - If doing extensive tuning, use cross-validation

4. **Don't cherry-pick results**
   - Report all metrics, not just the good ones
   - Show both successes and failures

---

## Verification Checklist

Before running validation, verify:

- [ ] `split_data.py` was run to create train/validation split
- [ ] `train.jsonl` has 2,700 samples
- [ ] `validation.jsonl` has 300 samples
- [ ] Training used only `train.jsonl`
- [ ] Validation loads from `validation.jsonl`
- [ ] No overlap between train and validation sets

---

## Advanced: Computing Aggregate Metrics

For a more comprehensive evaluation across all validation samples:

```python
# Compute metrics across all validation samples
total_correct = 0
total_expected = 0
total_predicted = 0

for sample in validation_data:
    prediction = generate_response(sample['prompt'])
    
    expected = set(sample['completion'].split('\n'))
    predicted = set(prediction.split('\n'))
    
    correct = expected & predicted
    
    total_correct += len(correct)
    total_expected += len(expected)
    total_predicted += len(predicted)

# Aggregate metrics
precision = total_correct / total_predicted if total_predicted > 0 else 0
recall = total_correct / total_expected if total_expected > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1_score:.2%}")
```

---

## Summary

✅ **Our validation strategy ensures honest evaluation by:**
1. Using a held-out validation set (300 samples, 10% of data)
2. Never exposing validation data during training
3. Testing on truly unseen examples
4. Computing accurate performance metrics
5. Avoiding overfitting and memorization

This gives you confidence that your fine-tuned model will generalize to real-world medical texts, not just regurgitate training examples.
