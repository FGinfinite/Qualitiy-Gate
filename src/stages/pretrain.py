import os
from typing import Tuple

import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from src.data import encode_data, get_data_statistics, load_and_prepare_dataset
from src.models.select_moe import SelectMoeForCausalLM, register_select_moe
from src.training.full_rank_finetuning import (
    print_trainable_parameters,
    save_full_rank_weights,
    setup_full_rank_training,
)
from src.utils.logging_utils import setup_training_logging
from src.utils.tools import grab_gpu

# 路由权重相关的模式定义
ROUTING_PATTERNS = [
    "quality_gate",  # Quality gate parameters
    ".gate.weight",  # MoE gate weights
    "router",  # Any router-related parameters
]

# ---------------------------------------------------------------------------
# 模型修改和 PEFT 配置的辅助函数
# ---------------------------------------------------------------------------


def get_peft_config(cfg: DictConfig) -> LoraConfig:
    """
    根据配置生成 PEFT LoRA 配置。

    Args:
        cfg (DictConfig): 训练配置。

    Returns:
        LoraConfig: 生成的 LoRA 配置。
    """
    if cfg.training.peft_mode != "lora":
        raise ValueError(f"无效的 peft_mode: {cfg.training.peft_mode}。此函数仅为 'lora' 设计。")

    # 从配置中提取 LoRA 参数
    lora_config = cfg.training.lora
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=list(cfg.training.lora.target_modules),
        modules_to_save=[],
    )


# ---------------------------------------------------------------------------
# 核心数据加载和模型初始化（重构后）
# ---------------------------------------------------------------------------


def get_model_and_tokenizer(
    cfg: DictConfig,
) -> Tuple[SelectMoeForCausalLM, AutoTokenizer]:
    """
    加载预转换的 Select-MoE 模型和分词器。
    """
    # 注册 Select-MoE 模型
    register_select_moe()

    # 加载预转换的 Select-MoE 模型
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }
    model = SelectMoeForCausalLM.from_pretrained(cfg.selector_model.path, **model_kwargs)

    # 从训练配置中覆写损失函数参数
    if hasattr(cfg.training, "quality_loss_weight"):
        model.config.quality_loss_weight = cfg.training.quality_loss_weight
    if hasattr(cfg.training, "quality_gate_init_mean"):
        model.config.quality_gate_init_mean = cfg.training.quality_gate_init_mean
    if hasattr(cfg.training, "quality_gate_init_std"):
        model.config.quality_gate_init_std = cfg.training.quality_gate_init_std
    if hasattr(cfg.training, "trash_expert_mode"):
        model.config.trash_expert_mode = cfg.training.trash_expert_mode
    if hasattr(cfg.training, "enable_load_balancing"):
        model.config.enable_load_balancing = cfg.training.enable_load_balancing
    if hasattr(cfg.training, "output_router_logits"):
        model.config.output_router_logits = cfg.training.output_router_logits

    # 加载分词器（使用原始模型名称）
    tokenizer = AutoTokenizer.from_pretrained(cfg.selector_model.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# 主要预训练阶段
# ---------------------------------------------------------------------------


def pretrain(cfg: DictConfig) -> None:
    """
    使用预转换的 Select-MoE 模型进行训练，并使用 PEFT 进行微调。
    """
    # 设置训练日志系统
    log, hydra_callback = setup_training_logging(__name__)

    log.info("--- 开始阶段 1：Select-MoE 预训练 ---")

    accelerator = Accelerator()
    if cfg.training.gpu_grab.grab:
        # 1. 抢占GPU并持有占位符
        log.info("--- 正在抢占GPU显存 ---")
        placeholders = grab_gpu(
            memory_need=cfg.training.gpu_grab.memory_need_gb,
            accelerator=accelerator,
            over_grab=cfg.training.gpu_grab.over_grab,
        )
        log.info("--- GPU显存抢占完成 ---")

    # 2. 加载 Select-MoE 模型和分词器
    log.info("正在加载和配置 Select-MoE 模型...")
    model, tokenizer = get_model_and_tokenizer(cfg)
    log.info("模型加载完成。")

    # 2. 配置微调模式
    log.info(f"正在为 '{cfg.training.peft_mode}' 模式配置微调...")

    if cfg.training.peft_mode == "lora":
        peft_config = get_peft_config(cfg)
        model = get_peft_model(model, peft_config)
        log.info("PEFT 模型已创建。可训练参数：")
        model.print_trainable_parameters()
    elif cfg.training.peft_mode == "full_rank":
        # 全秩微调模式：只微调路由参数（quality gates + MoE gates），但是全秩训练
        setup_full_rank_training(model, ROUTING_PATTERNS, mode="parameter")
        print_trainable_parameters(model)
    else:
        raise ValueError(f"不支持的微调模式: {cfg.training.peft_mode}。支持的模式: ['lora', 'full_rank']")

    # 确保 `use_cache` 已禁用以保证训练兼容性
    model.config.use_cache = False

    # 4. 加载并准备数据集
    dataset = load_and_prepare_dataset(cfg)

    # 5. 编码数据集
    tokenized_dataset = encode_data(
        dataset,
        tokenizer,
        max_seq_length=cfg.dataset.max_sequence_length,
        processing_num_workers=getattr(cfg.dataset, "processing_num_workers", 10),
        overwrite_cache=getattr(cfg.dataset, "overwrite_cache", False),
    )

    # 6. 输出数据统计信息
    get_data_statistics(tokenized_dataset)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    # 7. 配置训练参数
    # LoRA模式：保存中间权重；全秩微调和routing_only：不保存中间权重
    save_strategy = "epoch" if cfg.training.peft_mode == "lora" else "no"

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        logging_dir=f"{cfg.output_dir}/logs",
        logging_steps=10,
        save_strategy=save_strategy,
        report_to="none",
        # 改进的日志配置，确保与Hydra日志系统集成
        logging_first_step=True,
        log_level="info",
        log_level_replica="warning",
        disable_tqdm=False,  # 保持进度条显示
        logging_nan_inf_filter=True,  # 过滤NaN/Inf值
    )

    # 8. 初始化并运行训练器
    # 在初始化Trainer之前，释放占位符以提供干净的显存

    if cfg.training.gpu_grab.grab:
        del placeholders
        torch.cuda.empty_cache()
        if accelerator.is_main_process:
            log.info("已释放GPU占位符以准备训练...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[hydra_callback],  # 使用设置好的日志回调
    )

    log.info("正在使用 Hugging Face Trainer 开始训练...")
    # 使用 try...finally 确保训练过程的稳健性
    try:
        trainer.train()
    finally:
        # 可以在此处添加清理代码（如果需要）
        pass
    log.info("训练完成。")

    # 9. 保存最终模型
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if cfg.training.peft_mode == "lora":
            log.info(f"正在将最终的 PEFT 适配模型保存到 {cfg.output_dir}")
            trainer.save_model(cfg.output_dir)
        elif cfg.training.peft_mode == "full_rank":
            log.info(f"正在将路由全秩微调权重保存到 {cfg.output_dir}")
            full_rank_weights_path = os.path.join(cfg.output_dir, "full_rank_weights.pt")
            save_full_rank_weights(model, ROUTING_PATTERNS, full_rank_weights_path, mode="parameter")

    log.info("--- 阶段 1：预训练完成 ---")


if __name__ == "__main__":
    # 这部分用于独立执行。
    # 这里需要一个完整的配置对象。
    pass
