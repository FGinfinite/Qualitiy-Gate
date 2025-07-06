# src/stages/pretrain.py
from typing import Dict, Tuple

import torch
from accelerate import Accelerator
from datasets import Dataset, concatenate_datasets, load_dataset
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    OlmoeForCausalLM,
    Trainer,
    TrainingArguments,
)

from src.modeling import TrashCanMoEForCausalLM, replace_moe_layers_with_trashcan
from utils.tools import grab_gpu

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
# 核心数据加载和模型初始化（重构后）
# ---------------------------------------------------------------------------


def get_model_and_tokenizer(
    cfg: DictConfig,
) -> Tuple[TrashCanMoEForCausalLM, AutoTokenizer]:
    """
    遵循“先加载标准模型，后修改”的原则，加载并准备模型和分词器。
    """
    # 1. 加载一个标准的、未经修改的 OlmoeForCausalLM 模型
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }
    base_model = OlmoeForCausalLM.from_pretrained(
        cfg.selector_model.name, **model_kwargs
    )

    # 2. 将模型中的 MoE 层替换为我们的 TrashCanMoE 层
    replace_moe_layers_with_trashcan(base_model, base_model.config, cfg)

    # 3. 将修改过的模型包装到 TrashCanMoEForCausalLM 中
    # 这个包装器包含了我们自定义的、基于挂钩的 forward 方法
    loss_params = {
        "constraint_loss_weight": cfg.training.constraint_loss_weight,
        "trash_can_loss_beta": cfg.training.trash_can_loss_beta,
    }
    final_model = TrashCanMoEForCausalLM(base_model.config, **loss_params)

    # 手动将修改过的模型组件（model 和 lm_head）赋给我们的包装器
    final_model.model = base_model.model
    final_model.lm_head = base_model.lm_head

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(cfg.selector_model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return final_model, tokenizer


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


def tokenize_function(example: Dict, tokenizer: AutoTokenizer, cfg: DictConfig) -> Dict:
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
        max_length=cfg.dataset.max_sequence_length,
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
    if cfg.training.gpu_grab.grab:
        # 1. 抢占GPU并持有占位符
        print("--- 正在抢占GPU显存 ---")
        placeholders = grab_gpu(
            memory_need=cfg.training.gpu_grab.memory_need_gb,
            accelerator=accelerator,
            over_grab=cfg.training.gpu_grab.over_grab,
        )
        print("--- GPU显存抢占完成 ---")

    # 2. 加载自定义模型和分词器
    # get_model_and_tokenizer 现在处理了所有复杂的设置
    print("正在加载和配置 MoE 模型...")
    model, tokenizer = get_model_and_tokenizer(cfg)
    print("模型加载和修改完成。")

    # 2. 配置 PEFT 进行微调
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
        batched=False,
        fn_kwargs={"tokenizer": tokenizer, "cfg": cfg},
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
    # 在初始化Trainer之前，释放占位符以提供干净的显存

    if cfg.training.gpu_grab.grab:
        del placeholders
        torch.cuda.empty_cache()
        if accelerator.is_main_process:
            print("已释放GPU占位符以准备训练...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("正在使用 Hugging Face Trainer 开始训练...")
    # 使用 try...finally 确保无论训练成功与否，钩子都能被正确移除
    try:
        # 在训练前激活前向挂钩
        model.activate_hooks()
        trainer.train()
    finally:
        # 训练结束后停用挂钩
        model.deactivate_hooks()
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
