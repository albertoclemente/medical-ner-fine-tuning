"""
Fine-tune Llama 3.2 3B Instruct for Medical NER with LoRA.
This script implements SFT (Supervised Fine-Tuning) with LoRA adaptation.
Tracks experiments with Weights & Biases.
"""

import json
import torch
import os
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
import wandb

# ========================================
# 1. CONFIGURATION
# ========================================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
HF_USERNAME = "your-username"  # CHANGE THIS to your HF username

# Generate timestamp for checkpoint naming
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
HF_MODEL_ID = f"{HF_USERNAME}/llama3-medical-ner-lora-{TIMESTAMP}"

print(f"Training session timestamp: {TIMESTAMP}")
print(f"HuggingFace model ID: {HF_MODEL_ID}")

# Initialize Weights & Biases
# Set WANDB_API_KEY environment variable or run 'wandb login' first
wandb.init(
    project="medical-ner-finetuning",
    name=f"llama3-medical-ner-{TIMESTAMP}",
    config={
        "model": MODEL_NAME,
        "lora_rank": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "epochs": 3,
        "batch_size": 16,  # effective batch size
    }
)
print("✓ Weights & Biases initialized")

# Login to Hugging Face (make sure HF_TOKEN is set)
if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))
    print("✓ Logged in to Hugging Face")
else:
    print("⚠ Warning: HF_TOKEN not found. Run 'huggingface-cli login' first")

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
    """Format the data into Llama 3 chat format."""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical NER expert. Extract the requested entities from medical texts accurately.<|eot_id|><|start_header_id|>user<|end_header_id|>

{sample['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{sample['completion']}<|eot_id|>"""

print("\n" + "="*80)
print("LOADING DATASETS")
print("="*80)

    # Load datasets
    print("\n📂 Loading datasets...")
    train_data = load_jsonl('../data/train.jsonl')
    val_data = load_jsonl('../data/validation.jsonl')

print(f"✓ Train samples loaded: {len(train_data)}")
print(f"✓ Validation samples loaded: {len(val_data)}")

# Format data
train_formatted = [{"text": format_instruction(sample)} for sample in train_data]
val_formatted = [{"text": format_instruction(sample)} for sample in val_data]

# Create HF datasets
train_dataset = Dataset.from_list(train_formatted)
val_dataset = Dataset.from_list(val_formatted)

print(f"✓ Datasets formatted and ready")

# ========================================
# 3. LOAD MODEL AND TOKENIZER
# ========================================

print("\n" + "="*80)
print("LOADING MODEL AND TOKENIZER")
print("="*80)

# Quantization config for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"✓ Using 4-bit quantization (NF4)")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right",
    add_eos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

print(f"✓ Tokenizer loaded: {MODEL_NAME}")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print(f"✓ Base model loaded: {MODEL_NAME}")

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

print(f"✓ Model prepared for k-bit training")

# ========================================
# 4. CONFIGURE LoRA
# ========================================

print("\n" + "="*80)
print("CONFIGURING LoRA")
print("="*80)

lora_config = LoraConfig(
    r=16,                          # LoRA rank
    lora_alpha=32,                 # LoRA alpha (scaling)
    target_modules=[               # Layers to apply LoRA
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,             # Dropout for regularization
    bias="none",                   # No bias training
    task_type="CAUSAL_LM"          # Causal language modeling
)

model = get_peft_model(model, lora_config)

print(f"✓ LoRA configuration applied:")
print(f"  - Rank (r): {lora_config.r}")
print(f"  - Alpha: {lora_config.lora_alpha}")
print(f"  - Dropout: {lora_config.lora_dropout}")
print(f"  - Target modules: {len(lora_config.target_modules)}")

print(f"\n✓ Trainable parameters:")
model.print_trainable_parameters()

# ========================================
# 5. TOKENIZE DATA
# ========================================

print("\n" + "="*80)
print("TOKENIZING DATASETS")
print("="*80)

def tokenize_function(examples):
    """Tokenize the texts."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding=False,
    )

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train set"
)

tokenized_val = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=val_dataset.column_names,
    desc="Tokenizing validation set"
)

print(f"✓ Train set tokenized: {len(tokenized_train)} samples")
print(f"✓ Validation set tokenized: {len(tokenized_val)} samples")

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ========================================
# 6. TRAINING ARGUMENTS
# ========================================

print("\n" + "="*80)
print("CONFIGURING TRAINING")
print("="*80)

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
    fp16=True,  # Use bf16 if using A100/H100
    
    # Hugging Face Hub
    push_to_hub=True,
    hub_model_id=HF_MODEL_ID,
    hub_strategy="checkpoint",
    hub_private_repo=False,
    
    # Misc
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="wandb",  # Enable Weights & Biases logging
    run_name=f"llama3-medical-ner-{TIMESTAMP}",  # W&B run name
    seed=42,
)

print(f"✓ Training configuration:")
print(f"  - Epochs: {training_args.num_train_epochs}")
print(f"  - Batch size (per device): {training_args.per_device_train_batch_size}")
print(f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  - Learning rate: {training_args.learning_rate}")
print(f"  - Eval steps: {training_args.eval_steps}")
print(f"  - Save steps: {training_args.save_steps}")
print(f"  - Hub model ID: {HF_MODEL_ID}")

# ========================================
# 7. INITIALIZE TRAINER
# ========================================

print("\n" + "="*80)
print("INITIALIZING TRAINER")
print("="*80)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

print(f"✓ Trainer initialized")

# Calculate training steps
total_steps = (len(tokenized_train) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)) * training_args.num_train_epochs
print(f"✓ Expected training steps: ~{total_steps}")
print(f"✓ Expected checkpoints: ~{total_steps // training_args.save_steps}")

# ========================================
# 8. TRAIN
# ========================================

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
print("This may take 2-3 hours on A100 GPU...")
print()

trainer.train()

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

# ========================================
# 9. SAVE FINAL MODEL
# ========================================

print("\nSaving final model...")
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")

print(f"✓ Model saved to: ./final_model")

# ========================================
# 10. PUSH TO HUB
# ========================================

print("\nPushing to Hugging Face Hub...")
try:
    trainer.push_to_hub(commit_message="Training complete - final model")
    print(f"✓ Model pushed to: https://huggingface.co/{HF_MODEL_ID}")
except Exception as e:
    print(f"⚠ Failed to push to hub: {e}")
    print("  You can manually push later using: trainer.push_to_hub()")

print("\n" + "="*80)
print("ALL DONE! 🎉")
print("="*80)
print(f"\nYour model is ready at:")
print(f"  Local: ./final_model")
print(f"  Hub: https://huggingface.co/{HF_MODEL_ID}")
print(f"\nNext steps:")
print(f"  1. Run validate_model.py to test the model")
print(f"  2. Check training logs in ./logs")
print(f"  3. View your model on Hugging Face Hub")
