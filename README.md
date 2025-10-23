# Medical Named Entity Recognition (NER) Fine-Tuning

Fine-tuning Llama 3.2 3B Instruct for extracting chemicals, diseases, and their relationships from medical literature using LoRA (Low-Rank Adaptation).

## 🎯 Project Overview

This project demonstrates how to fine-tune a large language model (Llama 3.2 3B Instruct) to perform Named Entity Recognition (NER) on medical texts, specifically:

- **Chemical Extraction**: Identify drug names and chemical compounds
- **Disease Extraction**: Extract disease and condition names  
- **Relationship Extraction**: Find relationships between chemicals and diseases

**Key Features**:
- LoRA-based fine-tuning for memory efficiency (4-bit quantization)
- Comprehensive evaluation with precision, recall, and F1 scores
- 80/10/10 train/validation/test data split
- Integration with Weights & Biases for experiment tracking
- Automated HuggingFace Hub deployment

## 📊 Dataset

- **Source**: `both_rel_instruct_all.jsonl`
- **Size**: 3,000 medical text examples
- **Format**: JSON Lines with instruction-based prompts
- **Split**:
  - Training: 2,400 examples (80%)
  - Validation: 300 examples (10%)
  - Test: 300 examples (10%)

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install -r requirements.txt
```

### 1. Split the Data

```bash
python scripts/split_data.py
```

This creates:
- `data/train.jsonl` (2,400 examples)
- `data/validation.jsonl` (300 examples)
- `data/test.jsonl` (300 examples)

### 2. Train the Model

```bash
python scripts/train.py
```

Training configuration:
- **Base Model**: Llama 3.2 3B Instruct
- **Method**: LoRA (rank=16, alpha=32)
- **Quantization**: 4-bit (NF4)
- **Epochs**: 3
- **Batch Size**: 4 (effective batch size 16 with gradient accumulation)
- **Time**: ~2-3 hours on A100 GPU

### 3. Evaluate on Test Set

```bash
python scripts/validate_model.py
```

Metrics calculated:
- Accuracy
- Precision (per entity type)
- Recall (per entity type)
- F1 Score (macro-averaged)

## 📓 Jupyter Notebook

The main notebook `notebooks/Medical_NER_Fine_Tuning.ipynb` provides an interactive walkthrough:

1. **Data Preparation**: Loading and splitting data
2. **Model Setup**: Base model + LoRA configuration
3. **Training**: Fine-tuning with validation monitoring
4. **Evaluation**: Comprehensive metrics on test set
5. **Custom Testing**: Test on your own medical texts (Section 17)

## 🏗️ Project Structure

```
├── README.md                                  # This file
├── requirements.txt                           # Python dependencies
│
├── notebooks/
│   ├── Medical_NER_Fine_Tuning.ipynb         # Main training notebook
│   └── Chapter_10_Fine_Tuning_using_Cohere_for_Medical_Data.ipynb  # Original Cohere example
│
├── scripts/
│   ├── train.py                              # Standalone training script
│   ├── validate_model.py                     # Test set evaluation
│   └── split_data.py                         # Data splitting utility
│
├── data/
│   └── both_rel_instruct_all.jsonl          # Training data (3,000 examples)
│
└── docs/
    ├── QUICK_START.md                        # Getting started guide
    ├── FINE_TUNING_PLAN.md                  # Detailed training plan
    ├── VALIDATION_STRATEGY.md               # Why validation vs test sets
    ├── THREE_WAY_SPLIT_GUIDE.md             # Data splitting best practices
    ├── CHECKPOINT_NAMING.md                 # Model naming conventions
    ├── PRACTICAL_USE_CASES.md               # Real-world applications
    └── IMPLEMENTATION_SUMMARY.md            # Change log
```

## 🔧 Configuration

### LoRA Parameters
```python
lora_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,          # Alpha
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)
```

### Training Arguments
```python
training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="epoch"
)
```

## 📈 Expected Results

After 3 epochs of training:
- **Training Loss**: ~0.3-0.5
- **Validation Loss**: ~0.4-0.6
- **Test F1 Score**: 0.75-0.85 (depending on entity type)

## 🧪 Custom Testing (Section 17)

The notebook includes 5 comprehensive test cases:

1. **Chemical Extraction**: Test ability to extract drug names
2. **Disease Extraction**: Test disease/condition identification
3. **Basic Relationship**: Single chemical-disease relationship
4. **Multiple Relationships**: Complex scenarios with multiple pairs
5. **Comprehensive**: All entities + relationships together

## 💾 Model Deployment

Models are automatically pushed to HuggingFace Hub with timestamp-based naming:

```
llama-3.2-3b-medical-ner-YYYYMMDD_HHMMSS
```

Example: `llama-3.2-3b-medical-ner-20241023_143022`

## 📚 Documentation

- **[Quick Start Guide](docs/QUICK_START.md)**: Get up and running in 5 minutes
- **[Fine-Tuning Plan](docs/FINE_TUNING_PLAN.md)**: Detailed training methodology
- **[Validation Strategy](docs/VALIDATION_STRATEGY.md)**: Understanding train/val/test splits
- **[Practical Use Cases](docs/PRACTICAL_USE_CASES.md)**: Real-world applications

## ⚠️ Important Notes

1. **Validation Set**: Used ONLY during training for monitoring - never for final evaluation
2. **Test Set**: "Sacred" dataset - used only once after training is complete
3. **3 Epochs**: Optimal for LoRA fine-tuning (prevents overfitting, cost-effective)
4. **GPU Requirements**: ~16GB VRAM (A100, V100, or T4 recommended)

## 🔗 Resources

- **Llama 3.2**: [Meta AI](https://ai.meta.com/llama/)
- **LoRA Paper**: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- **HuggingFace PEFT**: [Documentation](https://huggingface.co/docs/peft)
- **Weights & Biases**: [wandb.ai](https://wandb.ai)

## 📝 License

This project is for educational purposes. Please ensure you comply with:
- Llama 3.2 license agreement
- Medical data usage regulations
- Appropriate disclaimers for medical applications

## 🤝 Contributing

This is a research/educational project. Feel free to:
- Report issues
- Suggest improvements
- Share your fine-tuning results

## 📧 Contact

For questions about this implementation, please open an issue in the repository.

---

**Built with**: PyTorch • HuggingFace Transformers • PEFT • BitsAndBytes • Weights & Biases
