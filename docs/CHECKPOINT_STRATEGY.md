# Checkpoint Strategy

## Overview

The training process has been configured to save and upload checkpoints **every 50 steps** to Hugging Face Hub with unique timestamped names.

## Configuration

### Checkpoint Frequency
- **Local saves**: Every 50 steps to `./llama3-medical-ner-lora/checkpoint-{step}/`
- **HF Hub uploads**: Every 50 steps (automatic via custom callback)
- **Evaluation**: Every 50 steps on validation set

### Checkpoint Naming Convention

Each checkpoint is uploaded to Hugging Face Hub with a unique timestamped identifier:

```
Format: {username}/llama3-medical-ner-checkpoint-{step}-{timestamp}

Examples:
- albyos/llama3-medical-ner-checkpoint-50-20251104_143022
- albyos/llama3-medical-ner-checkpoint-100-20251104_145511
- albyos/llama3-medical-ner-checkpoint-150-20251104_152003
```

### Final Model

The final trained model is uploaded separately with its own timestamp:

```
Format: {username}/llama3-medical-ner-lora-final-{timestamp}

Example:
- albyos/llama3-medical-ner-lora-final-20251104_161245
```

## Implementation Details

### Custom Callback

A custom `CheckpointUploadCallback` handles automatic uploads:

```python
class CheckpointUploadCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        # Get checkpoint directory
        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        
        # Create timestamped model ID
        checkpoint_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_model_id = f"{username}/llama3-medical-ner-checkpoint-{step}-{timestamp}"
        
        # Upload to HF Hub
        api.upload_folder(
            folder_path=checkpoint_dir,
            repo_id=checkpoint_model_id,
            commit_message=f"Checkpoint at step {state.global_step}"
        )
```

### TrainingArguments Configuration

```python
training_args = TrainingArguments(
    save_strategy="steps",
    save_steps=50,              # Checkpoint every 50 steps
    save_total_limit=None,      # Keep all checkpoints
    eval_strategy="steps",
    eval_steps=50,              # Evaluate every 50 steps
    push_to_hub=False,          # Custom callback handles uploads
    # ... other args
)
```

## Expected Checkpoints

Based on the training configuration:

- **Dataset**: 2,400 training samples
- **Batch size**: 4
- **Gradient accumulation**: 4
- **Effective batch size**: 16
- **Steps per epoch**: ~150 (2400 / 16)
- **Total epochs**: 3
- **Total steps**: ~450

### Expected checkpoint count: **~9 checkpoints**

Checkpoints will be created at steps:
- 50, 100, 150, 200, 250, 300, 350, 400, 450

## Monitoring

### During Training

The callback prints detailed information for each upload:

```
================================================================================
📤 Uploading checkpoint to Hugging Face Hub
   Step: 50
   Model ID: albyos/llama3-medical-ner-checkpoint-50-20251104_143022
================================================================================

✅ Checkpoint uploaded successfully!
   URL: https://huggingface.co/albyos/llama3-medical-ner-checkpoint-50-20251104_143022
```

### Weights & Biases

Checkpoint URLs are automatically logged to W&B:

```python
wandb.log({
    "checkpoint_step": state.global_step,
    "checkpoint_url": f"https://huggingface.co/{checkpoint_model_id}"
})
```

## Storage Considerations

### Local Storage
- Each checkpoint: ~3-4 GB (LoRA adapters + optimizer states)
- All checkpoints: ~27-36 GB
- Set `save_total_limit=3` if local disk space is limited

### Hugging Face Hub
- Each checkpoint is uploaded as a separate repository
- No storage limits on number of repositories
- Each checkpoint repo is independently versioned

## Best Practices

### 1. Checkpoint Selection
After training, review checkpoints based on:
- Validation loss (logged every 50 steps)
- W&B metrics dashboard
- Choose checkpoint with best validation performance

### 2. Cleanup
If needed, delete intermediate checkpoints:

```python
from huggingface_hub import delete_repo

# Delete a specific checkpoint
delete_repo("username/llama3-medical-ner-checkpoint-50-20251104_143022")
```

### 3. Recovery
If training is interrupted, resume from latest checkpoint:

```python
# Load specific checkpoint
model = PeftModel.from_pretrained(
    base_model,
    "username/llama3-medical-ner-checkpoint-400-20251104_160122"
)

# Continue training
trainer.train(resume_from_checkpoint="./llama3-medical-ner-lora/checkpoint-400")
```

## Troubleshooting

### Upload Failures

If a checkpoint upload fails:
1. Checkpoint is still saved locally at `./llama3-medical-ner-lora/checkpoint-{step}/`
2. Error message printed but training continues
3. Manually upload later using:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./llama3-medical-ner-lora/checkpoint-50",
    repo_id="username/llama3-medical-ner-checkpoint-50-{timestamp}",
    commit_message="Manual upload of checkpoint 50"
)
```

### Authentication Issues

Ensure HF_TOKEN is set correctly:

```python
import os
from huggingface_hub import login

os.environ["HF_TOKEN"] = "hf_your_token_here"
login(token=os.environ["HF_TOKEN"])
```

### Disk Space

Monitor local disk usage:

```bash
du -sh ./llama3-medical-ner-lora/checkpoint-*
```

Delete local checkpoints after successful upload:

```python
import shutil
shutil.rmtree("./llama3-medical-ner-lora/checkpoint-50")
```

## Summary

✅ **Checkpoints every 50 steps**  
✅ **Unique timestamped names for each checkpoint**  
✅ **Automatic upload to Hugging Face Hub**  
✅ **Full tracking in Weights & Biases**  
✅ **Easy recovery and checkpoint selection**  

This strategy ensures you have comprehensive checkpoints throughout training while maintaining organized, uniquely-named models on Hugging Face Hub.
