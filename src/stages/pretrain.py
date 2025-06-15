# src/stages/pretrain.py
from typing import Dict, Tuple

import torch
from accelerate import Accelerator
from datasets import Dataset, concatenate_datasets, load_dataset
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
    """Loads the model and tokenizer from Hugging Face with memory optimization."""
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
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


def load_and_prepare_dataset(cfg: DictConfig) -> Dataset:
    """Loads, prepares, and combines datasets from Hugging Face Hub."""
    # Load and concatenate datasets from the provided paths
    all_datasets = [load_dataset(path, split="train") for path in cfg.dataset.paths]
    dataset = concatenate_datasets(all_datasets)
    print(
        f"Loaded and concatenated datasets {cfg.dataset.paths} "
        f"with {len(dataset)} total samples."
    )

    # Shuffle the dataset
    dataset = dataset.shuffle(seed=cfg.dataset.seed)

    # Select a subset based on the ratio
    subset_size = int(len(dataset) * cfg.dataset.subset_ratio)
    dataset = dataset.select(range(subset_size))
    print(
        f"Selected a subset of {len(dataset)} samples "
        f"({cfg.dataset.subset_ratio * 100:.2f}% of the total)."
    )

    return dataset


def tokenize_function(example: Dict, tokenizer: AutoTokenizer) -> Dict:
    """
    Tokenizes a single example from the dataset.
    Handles different conversation formats ('conversations' vs. 'conversation')
    and cases where conversation data might be missing or None.
    """
    text_parts = []
    # Use .get() to safely access keys that might not exist and handle None values
    conversation_data = example.get("conversations") or example.get("conversation")

    if conversation_data:
        for turn in conversation_data:
            if isinstance(turn, dict):
                # Handle teknium/OpenHermes-2.5 format
                if "from" in turn and "value" in turn:
                    role = turn.get("from", "unknown")
                    content = turn.get("value", "")
                    text_parts.append(f"{role}: {content}")
                # Handle allenai/WildChat-1M format
                elif "role" in turn and "content" in turn:
                    role = turn.get("role", "unknown")
                    content = turn.get("content", "")
                    text_parts.append(f"{role}: {content}")

    formatted_text = "\n".join(text_parts)

    return tokenizer(
        formatted_text,
        truncation=True,
        padding="max_length",
        max_length=512,
    )


def pretrain(cfg: DictConfig) -> None:
    """
    Pre-trains the selector model using the Hugging Face Trainer API,
    fully compatible with Accelerate and FSDP.
    """
    print(
        "--- Starting Stage 1: Selector Model Pre-training (Trainer with Accelerate) ---"
    )

    # Initialize Accelerator. The Trainer will automatically use this instance.
    accelerator = Accelerator()

    # Model and Tokenizer
    model, tokenizer = get_model_and_tokenizer(cfg.selector_model.name)

    # The following block is specific to FSDP and should not be run in DDP mode.
    # It's commented out to prevent errors when not using FSDP.
    # if accelerator.state.fsdp_plugin is not None:
    #     accelerator.state.fsdp_plugin.model_init_kwargs = {"config": model.config}

    # Disable `use_cache` to be compatible with gradient checkpointing enabled in FSDP.
    model.config.use_cache = False

    # Freeze weights before passing the model to the Trainer
    freeze_non_router_weights(model)

    # Dataset and Data Collator
    dataset = load_and_prepare_dataset(cfg)
    # Determine original columns to remove after tokenization
    original_columns = dataset.column_names
    tokenized_dataset = dataset.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=original_columns,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training Arguments
    # These arguments are passed to the Trainer. Note that the Trainer will
    # automatically detect the Accelerate environment and configure itself for
    # distributed training (including FSDP).
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        logging_dir=f"{cfg.output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        # When using DDP, we don't need to worry about the FSDP-specific
        # activation checkpointing conflicts. The `use_cache=False` setting is
        # generally good practice when gradient checkpointing is used, so we keep it.
        report_to="none",  # Disable wandb/tensorboard reporting
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Start training
    print("Starting training with Hugging Face Trainer...")
    trainer.train()

    # Wait for all processes to finish before exiting
    accelerator.wait_for_everyone()

    # Explicitly save the final model in a format that's easy to load for inference
    if accelerator.is_main_process:
        print(f"Saving final model to {cfg.output_dir}")
        # This saves the consolidated model, making it easy to load with from_pretrained
        trainer.save_model(cfg.output_dir)

    print("\n--- Stage 1: Pre-training Completed ---")


if __name__ == "__main__":
    # This part is for standalone execution.
    pass
