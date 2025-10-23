# Fine-Tuning Plan for Medical NER Model

## Project Overview
Fine-tune an open-source LLM on medical named entity recognition (NER) and relationship extraction using the `both_rel_instruct_all.jsonl` dataset with SFT (Supervised Fine-Tuning) and LoRA (Low-Rank Adaptation).

---

## 1. Model Selection

### Recommended Model: **Llama-3.2-3B-Instruct**

**Rationale:**
- **Size**: 3B parameters - optimal balance between performance and resource efficiency
- **Instruction-tuned**: Already trained to follow instructions, perfect for our task format
- **Medical capability**: Strong base knowledge that transfers well to medical domains
- **LoRA compatibility**: Excellent support in PEFT library
- **Community support**: Extensive documentation and examples
- **License**: Open-source (Llama 3 Community License)

**Alternative options:**
- Mistral-7B-Instruct-v0.3 (if more capacity needed)
- Phi-3-mini-4k-instruct (if resource-constrained)

---

## 2. Dataset Preparation

### Current Dataset Statistics
- **Total examples**: 3,000
- **Format**: JSON Lines (.jsonl)
- **Tasks**: 
  - Chemical extraction (1,000 examples)
  - Disease extraction (1,000 examples)
  - Chemical-Disease relationships (1,000 examples)

### Train/Validation Split Strategy
```
Train set: 2,700 examples (90%)
Validation set: 300 examples (10%)

Per task:
- Chemical extraction: 900 train / 100 validation
- Disease extraction: 900 train / 100 validation
- Relationships: 900 train / 100 validation
```

### Implementation Steps

**Step 1: Split the dataset**
```python
import json
import random
from sklearn.model_selection import train_test_split

# Load data
with open('both_rel_instruct_all.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# Stratified split to maintain task distribution
random.seed(42)
train_data, val_data = train_test_split(
    data, 
    test_size=0.1, 
    random_state=42,
    shuffle=True
)

# Save splits
with open('train.jsonl', 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + '\n')

with open('validation.jsonl', 'w') as f:
    for item in val_data:
        f.write(json.dumps(item) + '\n')

print(f"Train samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
```

---

## 3. Environment Setup

### Required Libraries
```bash
pip install torch transformers datasets peft accelerate bitsandbytes
pip install huggingface-hub wandb trl
pip install scikit-learn
```

### Hardware Requirements
- **Minimum**: 1x GPU with 16GB VRAM (e.g., RTX 4090, V100)
- **Recommended**: 1x GPU with 24GB+ VRAM (e.g., A100, RTX 6000)
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB free space

---

## 4. LoRA Configuration

### Recommended LoRA Hyperparameters
```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,                          # LoRA rank
    lora_alpha=32,                 # LoRA alpha (scaling factor)
    target_modules=[               # Layers to apply LoRA
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,             # Dropout for LoRA layers
    bias="none",                   # Bias handling
    task_type=TaskType.CAUSAL_LM   # Task type
)
```

**Parameter Explanation:**
- `r=16`: Good balance between model capacity and training efficiency
- `lora_alpha=32`: 2x rank is a standard ratio
- `target_modules`: All attention and MLP layers for comprehensive adaptation
- `lora_dropout=0.05`: Regularization to prevent overfitting

---

## 5. Training Configuration

### Hyperparameters
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    # Output and logging
    output_dir="./llama3-medical-ner-lora",
    logging_dir="./logs",
    logging_steps=10,
    
    # Training parameters
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    
    # Optimization
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=100,
    
    # Checkpointing
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    
    # Memory optimization
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    
    # Mixed precision
    fp16=True,  # Use bf16 if using A100
    
    # Hugging Face Hub
    push_to_hub=True,
    hub_model_id="your-username/llama3-medical-ner-lora",
    hub_strategy="checkpoint",
    hub_private_repo=False,
    
    # Misc
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="wandb",  # Optional: for experiment tracking
)
```

### Checkpoint Upload Strategy
- **Frequency**: Every 100 steps
- **Location**: Hugging Face Hub
- **Retention**: Keep last 3 checkpoints locally
- **Best model**: Automatically tracked and saved
- **Naming**: Each training run uses a unique timestamp (format: `YYYYMMDD_HHMMSS`)
  - Example: `your-username/llama3-medical-ner-lora-20240115_143022`
  - This ensures each training session creates a separate repository on HuggingFace Hub
  - Prevents checkpoint conflicts between multiple training runs

---

## 6. Complete Training Script

Create `train.py`:

```python
import json
import torch
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
import os

# ========================================
# 1. CONFIGURATION
# ========================================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
HF_USERNAME = "your-username"  # CHANGE THIS

# Generate timestamp for checkpoint naming
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
HF_MODEL_ID = f"{HF_USERNAME}/llama3-medical-ner-lora-{TIMESTAMP}"

print(f"Training session: {TIMESTAMP}")
print(f"Model will be pushed to: {HF_MODEL_ID}")

# Login to Hugging Face
login(token=os.getenv("HF_TOKEN"))  # Set HF_TOKEN environment variable

# ========================================
# 2. LOAD DATA
# ========================================

def load_jsonl(file_path):
    """Load JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_instruction(sample):
    """Format the data into instruction format."""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical NER expert. Extract the requested entities from medical texts accurately.<|eot_id|><|start_header_id|>user<|end_header_id|>

{sample['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{sample['completion']}<|eot_id|>"""

# Load datasets
print("Loading datasets...")
train_data = load_jsonl('train.jsonl')
val_data = load_jsonl('validation.jsonl')

# Format data
train_formatted = [{"text": format_instruction(sample)} for sample in train_data]
val_formatted = [{"text": format_instruction(sample)} for sample in val_data]

# Create HF datasets
train_dataset = Dataset.from_list(train_formatted)
val_dataset = Dataset.from_list(val_formatted)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# ========================================
# 3. LOAD MODEL AND TOKENIZER
# ========================================

print("Loading model and tokenizer...")

# Quantization config for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right",
    add_eos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# ========================================
# 4. CONFIGURE LoRA
# ========================================

print("Configuring LoRA...")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========================================
# 5. TOKENIZE DATA
# ========================================

def tokenize_function(examples):
    """Tokenize the texts."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding=False,
    )

print("Tokenizing datasets...")
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)
tokenized_val = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=val_dataset.column_names,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ========================================
# 6. TRAINING ARGUMENTS
# ========================================

training_args = TrainingArguments(
    output_dir="./llama3-medical-ner-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id=HF_MODEL_ID,
    hub_strategy="checkpoint",
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    fp16=True,
    report_to="none",  # Change to "wandb" if using W&B
)

# ========================================
# 7. INITIALIZE TRAINER
# ========================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

# ========================================
# 8. TRAIN
# ========================================

print("Starting training...")
trainer.train()

# ========================================
# 9. SAVE FINAL MODEL
# ========================================

print("Saving final model...")
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")

# Push to hub
print("Pushing to Hugging Face Hub...")
trainer.push_to_hub()

print("Training complete!")
```

---

## 7. Execution Steps

### Step 1: Prepare Environment
```bash
# Create project directory
cd /Users/alberto/projects/building_llms/ch_10_fine_tuning

# Install dependencies
pip install torch transformers datasets peft accelerate bitsandbytes huggingface-hub trl scikit-learn
```

### Step 2: Split Dataset
```bash
# Create and run the split script
python split_data.py
```

Create `split_data.py`:
```python
import json
import random
from sklearn.model_selection import train_test_split

# Load data
with open('both_rel_instruct_all.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# Split
random.seed(42)
train_data, val_data = train_test_split(
    data, 
    test_size=0.1, 
    random_state=42,
    shuffle=True
)

# Save
with open('train.jsonl', 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + '\n')

with open('validation.jsonl', 'w') as f:
    for item in val_data:
        f.write(json.dumps(item) + '\n')

print(f"✓ Train samples: {len(train_data)}")
print(f"✓ Validation samples: {len(val_data)}")
```

### Step 3: Set Up Hugging Face Authentication
```bash
# Set your Hugging Face token
export HF_TOKEN="your_hugging_face_token_here"

# Or login interactively
huggingface-cli login
```

### Step 4: Run Training
```bash
python train.py
```

### Step 5: Monitor Training
- Check local logs in `./logs` directory
- View checkpoints in `./llama3-medical-ner-lora`
- Monitor on Hugging Face Hub at: `https://huggingface.co/your-username/llama3-medical-ner-lora`

---

## 8. Validation and Testing

Create `validate_model.py`:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

# Load base model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "./final_model",  # or "your-username/llama3-medical-ner-lora"
)
model.eval()

def generate_response(prompt_text):
    """Generate a response for a given prompt."""
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical NER expert. Extract the requested entities from medical texts accurately.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response
    response = response.split("assistant\n\n")[-1]
    return response

# Test on validation samples
with open('validation.jsonl', 'r') as f:
    val_samples = [json.loads(line) for line in f]

# Test a few examples
print("=" * 80)
print("VALIDATION EXAMPLES")
print("=" * 80)

for i, sample in enumerate(val_samples[:3]):
    print(f"\n--- Example {i+1} ---")
    print(f"Prompt:\n{sample['prompt'][:200]}...")
    print(f"\nExpected:\n{sample['completion']}")
    print(f"\nGenerated:\n{generate_response(sample['prompt'])}")
    print("-" * 80)
```

---

## 9. Hugging Face Hub Upload Details

### What Gets Uploaded
1. **LoRA adapters** (much smaller than full model)
2. **Training configuration**
3. **Tokenizer files**
4. **Training metrics and logs**
5. **Model card (README.md)**

### Expected Checkpoint Locations
```
https://huggingface.co/your-username/llama3-medical-ner-lora
├── checkpoint-100/
├── checkpoint-200/
├── checkpoint-300/
├── ...
└── final_model/
```

### Model Card Template

Create `README.md`:
```markdown
---
license: llama3
language:
- en
tags:
- medical
- ner
- named-entity-recognition
- lora
- llama3
datasets:
- medical-ner-dataset
metrics:
- loss
---

# Llama 3.2 3B Medical NER (LoRA)

## Model Description
This is a LoRA fine-tuned version of Llama-3.2-3B-Instruct for medical named entity recognition and relationship extraction.

## Tasks
1. Chemical/drug entity extraction
2. Disease entity extraction  
3. Chemical-disease relationship extraction

## Training Details
- **Base Model**: meta-llama/Llama-3.2-3B-Instruct
- **Method**: LoRA (Low-Rank Adaptation)
- **Dataset**: 3,000 medical abstracts
- **Training Samples**: 2,700
- **Validation Samples**: 300
- **LoRA Rank**: 16
- **Training Steps**: ~2,025 (3 epochs)

## Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = PeftModel.from_pretrained(base_model, "your-username/llama3-medical-ner-lora")
```

## License
Llama 3 Community License
```

---

## 10. Expected Training Metrics

### Timeline (on A100 GPU)
- **Total training time**: ~2-3 hours
- **Steps per epoch**: ~675 (2,700 samples / batch size 4)
- **Total steps**: ~2,025 (3 epochs)
- **Checkpoints saved**: ~20 checkpoints

### Expected Performance
- **Initial eval loss**: ~2.5-3.0
- **Final eval loss**: ~0.5-1.0 (target)
- **Training loss**: Should decrease smoothly

---

## 11. Post-Training Tasks

### 1. Merge LoRA weights (optional)
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
peft_model = PeftModel.from_pretrained(base_model, "./final_model")

# Merge and save
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

### 2. Quantize for deployment (optional)
```python
# Quantize to GGUF for llama.cpp
# Or use bitsandbytes for 4-bit/8-bit inference
```

### 3. Create inference API
```python
# Deploy on Hugging Face Inference Endpoints
# Or create a local API with FastAPI
```

---

## 12. Troubleshooting

### Out of Memory (OOM)
**Solutions:**
- Reduce `per_device_train_batch_size` to 2 or 1
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing=True`
- Use 4-bit quantization (already enabled)

### Slow Training
**Solutions:**
- Ensure GPU is being used: `torch.cuda.is_available()`
- Check batch size isn't too small
- Enable `fp16` or `bf16`

### Model Not Learning
**Solutions:**
- Increase learning rate to 3e-4
- Increase LoRA rank to 32
- Train for more epochs
- Check data formatting

### Upload Failures
**Solutions:**
- Verify HF token: `huggingface-cli whoami`
- Check internet connection
- Reduce checkpoint frequency
- Use `hub_strategy="end"` to upload only at end

---

## 13. Cost Estimation

### Cloud GPU Costs (approximate)
- **Google Colab Pro**: $10/month (A100 access)
- **RunPod**: ~$0.50/hour (A40/A100)
- **Lambda Labs**: ~$1.10/hour (A100)

**Total estimated cost**: $2-5 for complete training

### Local GPU
- **Free** if you have compatible GPU
- **Electricity**: Negligible (~$0.50)

---

## 14. Checklist

- [ ] Install all dependencies
- [ ] Split dataset into train/validation
- [ ] Set up Hugging Face authentication
- [ ] Review and update `HF_USERNAME` in train.py
- [ ] Run training script
- [ ] Monitor training progress
- [ ] Validate model performance
- [ ] Upload final model to Hub
- [ ] Test inference on new examples
- [ ] Document results and model card

---

## Next Steps After Training

1. **Evaluate performance** on validation set
2. **Test on unseen medical texts**
3. **Compare with base model** (ablation study)
4. **Fine-tune hyperparameters** if needed
5. **Deploy** for production use
6. **Gather feedback** and iterate

---

## References

- [Llama 3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Training Guide](https://huggingface.co/docs/transformers/training)

---

**Good luck with your fine-tuning! 🚀**
