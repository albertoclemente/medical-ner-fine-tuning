# Checkpoint Naming with Timestamps

## Overview
Each training run automatically generates a unique timestamp that is appended to the HuggingFace model repository name. This ensures that:
- Multiple training runs don't overwrite each other
- Each training session is traceable
- Checkpoints are organized chronologically on HuggingFace Hub

## Timestamp Format
- **Format**: `YYYYMMDD_HHMMSS`
- **Example**: `20240115_143022` (January 15, 2024, 2:30:22 PM)

## How It Works

### 1. Automatic Generation
When you start training, the timestamp is generated automatically:

```python
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
HF_MODEL_ID = f"{HF_USERNAME}/llama3-medical-ner-lora-{TIMESTAMP}"
```

### 2. HuggingFace Repository Naming
Your model will be pushed to a unique repository:

```
your-username/llama3-medical-ner-lora-20240115_143022
```

### 3. Checkpoint Structure
During training, checkpoints are saved every 100 steps and automatically pushed to HuggingFace Hub:

```
HuggingFace Hub:
└── your-username/llama3-medical-ner-lora-20240115_143022/
    ├── checkpoint-100/
    ├── checkpoint-200/
    ├── checkpoint-300/
    └── ...
```

## Benefits

### 1. **Traceability**
- Know exactly when each training run started
- Easy to identify which checkpoint corresponds to which experiment

### 2. **No Conflicts**
- Multiple training runs won't overwrite each other
- Safe to run parallel experiments

### 3. **Version Control**
- Each training session creates a separate repository
- Easy to compare different training runs
- Rollback to previous versions if needed

### 4. **Organization**
- Chronological ordering on HuggingFace Hub
- Clear naming convention
- Easy to find specific training sessions

## Example Usage

### Training Session 1 (Morning)
```bash
# Started at 2024-01-15 09:30:00
# Model pushed to: your-username/llama3-medical-ner-lora-20240115_093000
```

### Training Session 2 (Afternoon)
```bash
# Started at 2024-01-15 14:30:22
# Model pushed to: your-username/llama3-medical-ner-lora-20240115_143022
```

### Training Session 3 (Next Day)
```bash
# Started at 2024-01-16 10:15:45
# Model pushed to: your-username/llama3-medical-ner-lora-20240116_101545
```

## Viewing Your Models

On HuggingFace Hub, you'll see:
```
https://huggingface.co/your-username/
├── llama3-medical-ner-lora-20240115_093000
├── llama3-medical-ner-lora-20240115_143022
├── llama3-medical-ner-lora-20240116_101545
└── ...
```

## Loading a Specific Checkpoint

To load a model from a specific training session:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# Load the specific training session
model = PeftModel.from_pretrained(
    base_model,
    "your-username/llama3-medical-ner-lora-20240115_143022"
)
```

## Customization

If you want to use a custom naming scheme instead of timestamps:

```python
# Option 1: Custom identifier
CUSTOM_ID = "experiment-v1"
HF_MODEL_ID = f"{HF_USERNAME}/llama3-medical-ner-lora-{CUSTOM_ID}"

# Option 2: Combine timestamp with custom name
EXPERIMENT_NAME = "high-lr-test"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
HF_MODEL_ID = f"{HF_USERNAME}/llama3-medical-ner-lora-{EXPERIMENT_NAME}-{TIMESTAMP}"
```

## Best Practices

1. **Keep the timestamp**: It provides automatic versioning
2. **Add meaningful prefixes** for experiments:
   ```python
   HF_MODEL_ID = f"{HF_USERNAME}/llama3-medical-ner-lora-{EXPERIMENT_TYPE}-{TIMESTAMP}"
   ```
3. **Document your training runs**: Keep a log of what each timestamp represents
4. **Clean up old models**: Delete unsuccessful training runs from HuggingFace Hub to save space

## Troubleshooting

### Issue: Model name too long
If your username + timestamp creates a name that's too long:
```python
# Use shorter format
TIMESTAMP = datetime.now().strftime("%m%d_%H%M")  # MMDD_HHMM
```

### Issue: Want to continue from a checkpoint
If you're resuming training, use the original model ID:
```python
# Don't create new timestamp, use existing one
HF_MODEL_ID = "your-username/llama3-medical-ner-lora-20240115_143022"
```

---

**Note**: This timestamp naming convention is automatically implemented in both `train.py` and `Medical_NER_Fine_Tuning.ipynb`.
