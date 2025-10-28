import pytest
import sys
from packaging import version

transformers = pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")
peft = pytest.importorskip("peft")
torch = pytest.importorskip("torch")

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.6.0"),
    reason="Requires PyTorch >= 2.6 for security compliance. Skip on older versions."
)
def test_tiny_model_trains_single_step(tmp_path):
    """Ensures the training stack (tokenizer → model → LoRA → Trainer) is functional."""
    model_name = "sshleifer/tiny-gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = Dataset.from_dict({"text": ["Medical NER test prompt."] * 4})

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=64,
        )

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # Required for gradient checkpointing-style training

    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=str(tmp_path / "outputs"),
        per_device_train_batch_size=2,
        num_train_epochs=1,
        max_steps=1,
        learning_rate=5e-4,
        logging_steps=1,
        evaluation_strategy="no",
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
        seed=42,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    assert trainer.state.global_step == 1, "Trainer did not complete the expected training step."
