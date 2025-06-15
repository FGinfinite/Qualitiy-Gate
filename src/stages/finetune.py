from typing import Dict, List

import torch
from accelerate import Accelerator
from datasets import load_dataset
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def train(cfg: DictConfig) -> None:
    """
    Main training function for fine-tuning the model with LoRA, using the
    Hugging Face Trainer for a more robust and standardized training loop.

    Args:
        cfg (DictConfig): Configuration object from Hydra.
    """
    set_seed(cfg.training.seed)
    # The Trainer will automatically detect and use the Accelerator instance.
    accelerator = Accelerator()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    # PEFT configuration for LoRA
    lora_config = LoraConfig(
        r=cfg.peft.lora_r,
        lora_alpha=cfg.peft.lora_alpha,
        target_modules=list(cfg.peft.target_modules),
        lora_dropout=cfg.peft.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and preprocess dataset
    raw_dataset = load_dataset("json", data_files=cfg.dataset.path)

    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
        # Assumes the JSONL file has a "text" field.
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg.training.max_length,
        )

    tokenized_dataset = raw_dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
    )

    # Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training Arguments
    # The output_dir is now managed by the launch script and Hydra.
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        learning_rate=cfg.training.learning_rate,
        logging_dir=cfg.training.logging_dir,
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.save_strategy,
        report_to=cfg.training.report_to,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,  # Pass tokenizer to save it with the model
    )

    # Start training
    print("Starting fine-tuning with Hugging Face Trainer...")
    trainer.train()

    # Wait for all processes to finish before saving
    accelerator.wait_for_everyone()

    # Save the final LoRA adapter
    if accelerator.is_main_process:
        print(f"Saving final LoRA adapter to {cfg.output_dir}")
        trainer.save_model(cfg.output_dir)

    print("\n--- Stage 3: Fine-tuning Completed ---")
