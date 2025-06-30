# src/stages/pretrain.py
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import Dataset, concatenate_datasets, load_dataset
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

from src.modeling import replace_moe_layers_with_trashcan


# ---------------------------------------------------------------------------
# 模型修改和 PEFT 配置的辅助函数
# ---------------------------------------------------------------------------


def get_peft_config(cfg: DictConfig) -> LoraConfig:
    """
    根据配置生成 PEFT LoRA 配置。
    """
    if cfg.training.peft_mode != "lora":
        raise ValueError(
            f"无效的 peft_mode: {cfg.training.peft_mode}。此函数仅为 'lora' 设计。"
        )

    # 从配置中提取 LoRA 参数
    lora_config = cfg.training.lora
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=list(lora_config.target_modules),
    )


# ---------------------------------------------------------------------------
# 核心数据加载和模型初始化（基本不变）
# ---------------------------------------------------------------------------


def get_model_and_tokenizer(
    model_name: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """从 Hugging Face 加载模型和分词器，并进行内存优化。"""
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_and_prepare_dataset(cfg: DictConfig) -> Dataset:
    """从 Hugging Face Hub 加载、准备和合并数据集。"""
    all_datasets = [load_dataset(path, split="train") for path in cfg.dataset.paths]
    dataset = concatenate_datasets(all_datasets)
    print(f"已加载并合并数据集 {cfg.dataset.paths} 总共有 {len(dataset)} 个样本。")
    dataset = dataset.shuffle(seed=cfg.dataset.seed)
    subset_size = int(len(dataset) * cfg.dataset.subset_ratio)
    dataset = dataset.select(range(subset_size))
    print(
        f"已选择 {len(dataset)} 个样本的子集 "
        f"({cfg.dataset.subset_ratio * 100:.2f}% 的总量)。"
    )
    return dataset


def tokenize_function(example: Dict, tokenizer: AutoTokenizer) -> Dict:
    """对数据集中的单个样本进行分词。"""
    text_parts = []
    conversation_data = example.get("conversations") or example.get("conversation")
    if conversation_data:
        for turn in conversation_data:
            if isinstance(turn, dict):
                role = turn.get("from") or turn.get("role", "unknown")
                content = turn.get("value") or turn.get("content", "")
                text_parts.append(f"{role}: {content}")
    formatted_text = "\n".join(text_parts)
    return tokenizer(
        formatted_text,
        truncation=True,
        padding="max_length",
        max_length=512,
    )


# ---------------------------------------------------------------------------
# 主要预训练阶段
# ---------------------------------------------------------------------------


def pretrain(cfg: DictConfig) -> None:
    """
    通过将标准 MoE 层替换为自定义的 'TrashCanMoE' 层，并使用 PEFT 进行微调，
    对 MoE 模型执行高级预训练。
    """
    print("--- 开始阶段 1：高级 MoE 预训练 ---")

    accelerator = Accelerator()

    # 1. 加载模型和分词器
    model, tokenizer = get_model_and_tokenizer(cfg.selector_model.name)

    # 2. 将原始 MoE 层替换为我们的自定义 TrashCanMoE 层
    print("正在将 MoE 层替换为 TrashCanMoE...")
    replace_moe_layers_with_trashcan(model, model.config)
    print("MoE 层替换完成。")

    # 3. 配置 PEFT 进行微调
    # 这取代了旧的 `freeze_non_router_weights` 函数。
    print(f"正在为 '{cfg.training.peft_mode}' 模式配置 PEFT...")
    peft_config = get_peft_config(cfg)
    model = get_peft_model(model, peft_config)
    print("PEFT 模型已创建。可训练参数：")
    model.print_trainable_parameters()

    # 确保 `use_cache` 已禁用以保证训练兼容性
    model.config.use_cache = False

    # 4. 加载并准备数据集
    dataset = load_and_prepare_dataset(cfg)
    original_columns = dataset.column_names
    tokenized_dataset = dataset.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=original_columns,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. 配置训练参数
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        logging_dir=f"{cfg.output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    # 6. 初始化并运行训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("正在使用 Hugging Face Trainer 开始训练...")
    trainer.train()
    print("训练完成。")

    # 7. 保存最终模型
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"正在将最终的 PEFT 适配模型保存到 {cfg.output_dir}")
        trainer.save_model(cfg.output_dir)

    print("\n--- 阶段 1：预训练完成 ---")


if __name__ == "__main__":
    # 这部分用于独立执行。
    # 这里需要一个完整的配置对象。
    pass
