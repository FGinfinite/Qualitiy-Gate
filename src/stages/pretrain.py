# src/stages/pretrain.py
import os
from typing import Dict, Tuple

import hydra
import torch
from datasets import Dataset
from omegaconf import DictConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def get_model_and_tokenizer(
    model_name: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Loads the model and tokenizer from Hugging Face."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def freeze_non_router_weights(model: AutoModelForCausalLM) -> None:
    """Freezes all weights except for the router/gate layers."""
    for name, param in model.named_parameters():
        if "router" in name or "gate" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    print("Froze non-router weights. Training only router/gate layers.")


def create_dummy_dataset(num_samples: int = 100) -> Dataset:
    """Creates a dummy dataset for testing purposes."""
    data = {"text": [f"This is a dummy sentence {i}" for i in range(num_samples)]}
    return Dataset.from_dict(data)


def tokenize_function(examples: Dict, tokenizer: AutoTokenizer) -> Dict:
    """Tokenizes the text in the dataset."""
    return tokenizer(examples["text"], truncation=True, padding="max_length")


def pretrain(cfg: DictConfig) -> None:
    """
    Pre-trains the selector model using the Hugging Face Trainer API.
    """
    print("--- Starting Stage 1: Selector Model Pre-training (HF Trainer) ---")

    # Model and Tokenizer
    model, tokenizer = get_model_and_tokenizer(cfg.selector_model.name)

    # Freeze weights
    freeze_non_router_weights(model)

    # Dataset
    dummy_dataset = create_dummy_dataset()
    tokenized_dataset = dummy_dataset.map(
        tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer}
    )
    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask"]
    )

    # Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=cfg.output.model_save_path,
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        save_strategy="epoch",  # Save at the end of each epoch
        logging_dir="./logs",
        logging_steps=10,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Start Training
    print("Starting training...")
    trainer.train()
    print("\n--- Stage 1: Pre-training Completed ---")


if __name__ == "__main__":
    pretrain()