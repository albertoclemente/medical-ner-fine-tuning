# Three-Way Data Split Guide

## Overview

This project uses a **proper machine learning data split** with three separate datasets:

```
Total Data: 3,000 samples
├── Train (80%): 2,400 samples → Used for fine-tuning
├── Validation (10%): 300 samples → Monitored during training (W&B)
└── Test (10%): 300 samples → Used ONLY for final evaluation
```

---

## Why Three Splits?

### ❌ Common Mistake: Two Splits Only
Many tutorials use only train/validation:
- Train (90%) - for training
- Validation (10%) - for both monitoring AND final evaluation

**Problem**: If you tune your model based on validation performance, the validation set is no longer "unseen"!

### ✅ Correct Approach: Three Splits
```
1. Train Set (80%)
   - Purpose: Fine-tune the model
   - Used: During training iterations
   
2. Validation Set (10%)
   - Purpose: Monitor training progress
   - Used: During training to track loss, detect overfitting
   - Shown in: Weights & Biases dashboard
   
3. Test Set (10%)
   - Purpose: Final evaluation
   - Used: ONLY ONCE after training completes
   - Ensures: True measure of generalization
```

---

## Data Split Process

### Step 1: Run split_data.py

```bash
python split_data.py
```

**What it does:**
```python
# First split: 80% train, 20% temp
train_data, temp_data = train_test_split(data, test_size=0.2)

# Second split: split the 20% into 10% val, 10% test  
val_data, test_data = train_test_split(temp_data, test_size=0.5)
```

**Output files:**
- `train.jsonl` - 2,400 samples
- `validation.jsonl` - 300 samples
- `test.jsonl` - 300 samples

---

## During Training

### What Happens with Each Dataset

**Train Set (`train.jsonl`)**:
- Model sees these examples during fine-tuning
- Weights are updated based on these samples
- Used: Every training step

**Validation Set (`validation.jsonl`)**:
- Model evaluates on these every 100 steps
- Validation loss is calculated
- Logged to Weights & Biases
- Used to: Detect overfitting, select best checkpoint
- NOT used to: Update model weights

**Test Set (`test.jsonl`)**:
- NOT touched during training
- NOT used for monitoring
- Stays completely unseen

---

## Weights & Biases Integration

### What You'll See in W&B

When training runs, W&B dashboard shows:

```
Charts:
├── train_loss (computed on training set)
├── eval_loss (computed on validation set) 👈 THIS IS KEY
├── learning_rate
├── epoch
└── step
```

**Key Metric**: `eval_loss`
- Calculated on the validation set every 100 steps
- Shows how well model generalizes during training
- If eval_loss increases while train_loss decreases → overfitting!

### W&B Setup

**In train.py:**
```python
import wandb

# Initialize W&B
wandb.init(
    project="medical-ner-finetuning",
    name=f"llama3-medical-ner-{TIMESTAMP}",
    config={...}
)

# In TrainingArguments
training_args = TrainingArguments(
    ...
    eval_strategy="steps",
    eval_steps=100,  # Evaluate every 100 steps
    report_to="wandb",  # Send metrics to W&B
    ...
)
```

**View results:**
- Dashboard: https://wandb.ai
- Project: medical-ner-finetuning
- Metrics updated in real-time during training

---

##After Training: Final Evaluation

### On Test Set ONLY

**After training completes**, run final evaluation:

```bash
python validate_model.py
```

**What it does:**
```python
# Load test set (completely unseen)
with open('test.jsonl', 'r') as f:
    test_samples = [json.loads(line) for line in f]

# Evaluate model
for sample in test_samples:
    prediction = model.generate(sample['prompt'])
    compare(prediction, sample['completion'])
```

**This is your TRUE performance metric!**
- Model has never seen these examples
- Not used for training or monitoring
- Honest assessment of generalization

---

## Complete Workflow

### 1. Data Preparation
```bash
# Split data into train/val/test
python split_data.py

# Output:
# ✓ train.jsonl (2,400 samples - 80%)
# ✓ validation.jsonl (300 samples - 10%)
# ✓ test.jsonl (300 samples - 10%)
```

### 2. Training with W&B Monitoring
```bash
# Set up W&B (one-time)
wandb login

# Start training
python train.py

# During training:
# - Model trains on train.jsonl
# - Validates on validation.jsonl every 100 steps
# - Metrics logged to W&B dashboard
# - Best model saved based on validation loss
```

### 3. Monitor Training (W&B Dashboard)
```
Open: https://wandb.ai

Watch:
- Training loss (should decrease)
- Validation loss (should decrease)
- If validation loss plateaus → training done
- If validation loss increases → overfitting!
```

### 4. Final Evaluation (Test Set)
```bash
# After training completes
python validate_model.py

# Evaluates on test.jsonl (completely unseen)
# This is your REAL performance!
```

---

## File Usage Summary

| File | Training | Monitoring | Final Eval |
|------|----------|------------|-----------|
| `train.jsonl` | ✅ Used | ❌ Not used | ❌ Not used |
| `validation.jsonl` | ❌ Not used | ✅ Used | ❌ Not used |
| `test.jsonl` | ❌ Not used | ❌ Not used | ✅ Used |

**Key Principle**: Each dataset has ONE purpose, used at ONE stage!

---

## Validation Loss in W&B

### What is it?
- Model's loss computed on the **validation set**
- Calculated every 100 training steps
- Does NOT affect model weights

### Why monitor it?
```python
if validation_loss decreases:
    # Model is learning to generalize ✅
    
if validation_loss plateaus:
    # Training is done, model saturated ⏸
    
if validation_loss increases (while train_loss decreases):
    # Overfitting! Model memorizing training data ⚠️
```

### Example W&B Chart
```
Step    Train Loss    Val Loss    Status
100     2.5           2.6         ✅ Good
200     2.2           2.3         ✅ Good
300     1.9           2.0         ✅ Good
400     1.7           1.8         ✅ Good
500     1.5           1.9         ⚠️  Val loss increased!
600     1.3           2.1         ❌ Overfitting detected
```

**Action**: Stop training around step 400-500 (when val loss was lowest)

---

## Best Practices

### ✅ DO:
1. **Run split_data.py ONCE before training**
   - Creates three separate files
   - Fixed random seed ensures reproducibility

2. **Use validation.jsonl for monitoring**
   - Watch validation loss in W&B
   - Stop training when validation loss plateaus

3. **Use test.jsonl ONLY ONCE**
   - After training is completely done
   - Report these metrics in papers/reports

4. **Never mix the datasets**
   - Don't train on validation data
   - Don't tune based on test performance

### ❌ DON'T:
1. **Don't test on training data**
   ```python
   # ❌ BAD
   evaluate_model(train_data)
   ```

2. **Don't reuse validation for final eval**
   ```python
   # ❌ BAD
   # You saw validation metrics during training!
   final_eval(validation_data)  
   ```

3. **Don't test multiple times**
   ```python
   # ❌ BAD
   # Testing multiple times = tuning to test set!
   for config in configurations:
       test_and_compare(test_data)
   ```

4. **Don't skip W&B logging**
   - You need validation metrics to detect overfitting
   - W&B makes this easy and visual

---

## Quick Reference

### Files Created
```
both_rel_instruct_all.jsonl  # Original data (3,000 samples)
├── train.jsonl              # 2,400 samples (80%)
├── validation.jsonl         # 300 samples (10%)
└── test.jsonl               # 300 samples (10%)
```

### Scripts
```
split_data.py          # Creates the 3-way split
train.py               # Trains model (uses train + val)
validate_model.py      # Final evaluation (uses test)
```

### W&B Project
```
Project: medical-ner-finetuning
URL: https://wandb.ai/<your-username>/medical-ner-finetuning
Metrics: train_loss, eval_loss, learning_rate, etc.
```

---

## Troubleshooting

### Q: I forgot to run split_data.py
**A:** Run it now:
```bash
python split_data.py
# This creates train.jsonl, validation.jsonl, test.jsonl
```

### Q: Can I change the split ratios?
**A:** Yes, edit `split_data.py`:
```python
# For 70/15/15 split:
train_data, temp_data = train_test_split(data, test_size=0.3)  # 70/30
val_data, test_data = train_test_split(temp_data, test_size=0.5)  # 15/15
```

### Q: W&B shows validation loss but I want to see test loss
**A:** No! Test set should ONLY be evaluated once after training. W&B shows validation loss (which is what you want to monitor).

### Q: Validation loss is increasing, what do I do?
**A:** Stop training! Your model is overfitting. Use an earlier checkpoint where validation loss was lowest.

---

## Summary

**The Golden Rule:**
```
Train Set    → Train the model
Val Set      → Monitor the model (W&B)
Test Set     → Evaluate the model (final, once)
```

This ensures:
- ✅ Honest performance metrics
- ✅ Early detection of overfitting
- ✅ Reproducible results
- ✅ Publishable findings

**Your test set performance is the number that matters!**
