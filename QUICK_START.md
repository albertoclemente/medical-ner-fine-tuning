# Quick Start Guide - Medical NER Fine-Tuning

## Prerequisites
- Python 3.10+
- CUDA-capable GPU (16GB+ VRAM recommended)
- Hugging Face account

## Step-by-Step Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Hugging Face Authentication
```bash
# Option A: Using environment variable
export HF_TOKEN="your_hugging_face_token_here"

# Option B: Using CLI (interactive)
huggingface-cli login
```

**Get your token from:** https://huggingface.co/settings/tokens

### 3. Update Configuration
Edit `train.py` and change:
```python
HF_USERNAME = "your-username"  # CHANGE THIS to your HF username
```

**Note on Model Naming**: Each training run automatically generates a unique timestamp that's added to your model name on HuggingFace Hub. For example:
- First run: `your-username/llama3-medical-ner-lora-20240115_143022`
- Second run: `your-username/llama3-medical-ner-lora-20240115_163015`

This prevents training runs from overwriting each other. See `CHECKPOINT_NAMING.md` for details.

### 4. Split the Dataset
```bash
python split_data.py
```

Expected output:
```
✓ Successfully split dataset:
  - Train samples: 2400 (80.0%)
  - Validation samples: 300 (10.0%) - for training monitoring
  - Test samples: 300 (10.0%) - for final evaluation

📊 Dataset usage:
  - Train: Used for fine-tuning
  - Validation: Monitored during training (shown in W&B)
  - Test: Used ONLY after training for final evaluation
```

**Important**: This creates 3 files:
- `train.jsonl` - used for training
- `validation.jsonl` - used for monitoring during training (W&B)
- `test.jsonl` - used ONLY for final evaluation after training

### 4b. Set Up Weights & Biases
```bash
# One-time setup
wandb login

# Follow prompts and paste your API key
# Get key from: https://wandb.ai/authorize
```

This enables real-time monitoring of:
- Training loss
- **Validation loss** (key metric for detecting overfitting)
- Learning rate
- GPU metrics

### 5. Start Training
```bash
python train.py
```

Training will:
- Load Llama 3.2 3B Instruct with 4-bit quantization
- Apply LoRA adapters
- Train for 3 epochs (~2-3 hours on A100)
- Save checkpoints every 100 steps
- Upload checkpoints to Hugging Face Hub
- **Log metrics to Weights & Biases**
- Monitor validation loss to detect overfitting
- Save final model to `./final_model`

**Monitor training:**
- Open W&B dashboard: https://wandb.ai
- Watch validation loss - if it increases, model is overfitting!
- Training stops when validation loss plateaus

### 6. Monitor Progress
Watch the training logs for:
- Training loss (should decrease)
- Eval loss (should decrease)
- Learning rate schedule
- GPU memory usage

### 7. Final Evaluation on Test Set
```bash
python validate_model.py
```

This will:
- Load the fine-tuned model
- Test on **unseen test samples** (never used during training or monitoring)
- Show expected vs actual outputs
- Compute accuracy metrics
- Test on custom examples

**Important**: This evaluates on `test.jsonl` (300 completely unseen samples). This gives you the TRUE performance metric since these samples were never used for training OR validation monitoring. See `THREE_WAY_SPLIT_GUIDE.md` for details.

## Expected Timeline

| Stage | Time (A100) | Time (RTX 4090) |
|-------|-------------|-----------------|
| Setup | 5 min | 5 min |
| Data prep | 1 min | 1 min |
| Model loading | 2 min | 3 min |
| Training | 2-3 hours | 4-5 hours |
| Validation | 5 min | 10 min |
| **Total** | **~3 hours** | **~5 hours** |

## Troubleshooting

### Out of Memory
Reduce batch size in `train.py`:
```python
per_device_train_batch_size=2,  # Change from 4 to 2
gradient_accumulation_steps=8,  # Change from 4 to 8
```

### Slow Training
Enable more aggressive mixed precision:
```python
bf16=True,  # If using Ampere GPU or newer
fp16=False,
```

### Upload Failures
Check your token:
```bash
huggingface-cli whoami
```

## File Structure
```
ch_10_fine_tuning/
├── both_rel_instruct_all.jsonl  # Original dataset (3000 examples)
├── train.jsonl                   # Training set (2700 examples)
├── validation.jsonl              # Validation set (300 examples)
├── train.py                      # Main training script
├── validate_model.py             # Validation script
├── split_data.py                 # Data splitting utility
├── requirements.txt              # Python dependencies
├── FINE_TUNING_PLAN.md          # Detailed documentation
├── QUICK_START.md               # This file
├── llama3-medical-ner-lora/     # Training output (created)
│   ├── checkpoint-100/
│   ├── checkpoint-200/
│   └── ...
├── final_model/                  # Final LoRA adapter (created)
└── logs/                         # Training logs (created)
```

## Next Steps After Training

1. **Test the model** on your own medical texts
2. **Compare performance** with base model
3. **Deploy** via Hugging Face Inference API
4. **Share** your model with the community

## Useful Commands

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check model on Hugging Face
# Visit: https://huggingface.co/your-username/llama3-medical-ner-lora

# Download model for offline use
huggingface-cli download your-username/llama3-medical-ner-lora
```

## Getting Help

- Check `FINE_TUNING_PLAN.md` for detailed information
- Review training logs in `./logs`
- Visit: https://huggingface.co/docs/transformers
- Visit: https://huggingface.co/docs/peft

---

**Happy Fine-Tuning! 🚀**
