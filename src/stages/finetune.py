# src/stages/finetune.py
import os
from typing import Dict

import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import get_peft_model, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

def finetune_target_model(cfg: DictConfig) -> None:
    """
    Fine-tunes the target model using the high-quality dataset selected in Stage 2.
    This process utilizes PEFT (LoRA) for efficient tuning and is launched via Accelerate.
    """
    print("--- Starting Stage 3: Target Model Fine-tuning ---")

    # Load Model and Tokenizer
    print(f"Loading target model: {cfg.target_model.name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.target_model.name,
        torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency
        use_cache=False # Recommended for training
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.target_model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # PEFT (LoRA) Configuration
    # Convert OmegaConf to a standard dict for PeftConfig
    peft_config_dict = OmegaConf.to_container(cfg.training.peft_config, resolve=True)
    peft_config = LoraConfig(**peft_config_dict)
    model = get_peft_model(model, peft_config)
    print("Applied PEFT (LoRA) configuration. Trainable parameters:")
    model.print_trainable_parameters()

    # Load Dataset from Stage 2
    print(f"Loading selected data from: {cfg.training.data_path}")
    dataset = load_dataset("json", data_files=cfg.training.data_path)["train"]

    def tokenize_function(examples: Dict) -> Dict:
        # Assumes the text is in a 'text' column, as produced by Stage 2.
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=cfg.training.max_length)

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=list(dataset.features) # Remove original columns
    )
    
    # Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training Arguments
    # Note: Many arguments are handled by Accelerate/DeepSpeed config now.
    # This TrainingArguments object is still needed by the Trainer.
    training_args = TrainingArguments(
        output_dir=cfg.output.adapter_save_path, # This is also the checkpoint dir
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        logging_dir=f"{cfg.output.adapter_save_path}/logs",
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        fp16=cfg.accelerate_config.use_fp16, # Sync with accelerate config
        bf16=cfg.accelerate_config.use_bf16, # Sync with accelerate config
        deepspeed=cfg.accelerate_config.deepspeed_config_path if cfg.accelerate_config.use_deepspeed else None,
    )

    # Trainer (Accelerate-aware)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Start Training
    print("Starting fine-tuning with LoRA...")
    trainer.train()
    
    # Save the final LoRA adapter
    print(f"Training complete. Saving LoRA adapter to: {cfg.output.adapter_save_path}")
    model.save_pretrained(cfg.output.adapter_save_path)

    print("\n--- Stage 3: Fine-tuning Completed ---")