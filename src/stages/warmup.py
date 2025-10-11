import logging
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
    set_seed,
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
# 批次大小验证函数
# ---------------------------------------------------------------------------


def validate_batch_size_configuration(total_batch_size: int, per_device_batch_size: int, world_size: int, log) -> int:
    """
    验证批次大小配置并计算梯度累积步数

    Args:
        total_batch_size: 目标总批次大小
        per_device_batch_size: 每设备批次大小
        world_size: 分布式训练的世界大小
        log: 日志记录器

    Returns:
        gradient_accumulation_steps: 梯度累积步数
    """
    effective_batch_size = per_device_batch_size * world_size

    if total_batch_size % effective_batch_size == 0:
        # 完全整除的情况
        gradient_accumulation_steps = total_batch_size // effective_batch_size
        actual_total_batch_size = total_batch_size
    else:
        # 无法整除的情况，选择最接近目标的配置
        target_grad_steps = total_batch_size / effective_batch_size
        grad_steps_floor = int(target_grad_steps)
        grad_steps_ceil = grad_steps_floor + 1

        # 计算两种选择的实际批次大小
        actual_batch_floor = grad_steps_floor * effective_batch_size
        actual_batch_ceil = grad_steps_ceil * effective_batch_size

        # 选择更接近目标的配置
        diff_floor = abs(total_batch_size - actual_batch_floor)
        diff_ceil = abs(total_batch_size - actual_batch_ceil)

        if diff_floor <= diff_ceil:
            gradient_accumulation_steps = grad_steps_floor
            actual_total_batch_size = actual_batch_floor
        else:
            gradient_accumulation_steps = grad_steps_ceil
            actual_total_batch_size = actual_batch_ceil

    log.info("批次大小配置验证完成:")
    log.info(f"  目标总批次大小: {total_batch_size}")
    log.info(f"  实际总批次大小: {actual_total_batch_size}")
    log.info(f"  每设备批次大小: {per_device_batch_size}")
    log.info(f"  世界大小: {world_size}")
    log.info(f"  有效批次大小: {effective_batch_size}")
    log.info(f"  梯度累积步数: {gradient_accumulation_steps}")

    if actual_total_batch_size != total_batch_size:
        log.warning(f"实际批次大小 ({actual_total_batch_size}) 与目标 ({total_batch_size}) 不同，差异: {abs(actual_total_batch_size - total_batch_size)}")

    return gradient_accumulation_steps


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

    # 新增的质量损失配置参数
    if hasattr(cfg.training, "quality_loss_type"):
        model.config.quality_loss_type = cfg.training.quality_loss_type
    if hasattr(cfg.training, "quality_loss_debug"):
        model.config.quality_loss_debug = cfg.training.quality_loss_debug

    # Beta moment matching参数
    if hasattr(cfg.training, "quality_loss_params"):
        params = cfg.training.quality_loss_params
        if hasattr(params, "beta_target_mean"):
            model.config.beta_target_mean = params.beta_target_mean
        if hasattr(params, "beta_target_var"):
            model.config.beta_target_var = params.beta_target_var
        if hasattr(params, "w_mean"):
            model.config.w_mean = params.w_mean
        if hasattr(params, "w_var"):
            model.config.w_var = params.w_var
        if hasattr(params, "lambda_var"):
            model.config.lambda_var = params.lambda_var
        if hasattr(params, "sample_wise_averaging"):
            model.config.sample_wise_averaging = params.sample_wise_averaging

    # 加载分词器（使用原始模型名称）
    tokenizer = AutoTokenizer.from_pretrained(cfg.selector_model.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 调整embedding大小
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        log = logging.getLogger(__name__)
        log.info(f"调整embedding大小: {embedding_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # 启用gradient checkpointing
    model.gradient_checkpointing_enable()
    log = logging.getLogger(__name__)
    log.info("已启用gradient checkpointing")

    # 启用输入梯度计算
    model.enable_input_require_grads()
    log.info("已启用输入梯度计算")

    return model, tokenizer


# ---------------------------------------------------------------------------
# 主要预训练阶段
# ---------------------------------------------------------------------------


def warmup(cfg: DictConfig) -> None:
    """
    使用预转换的 Select-MoE 模型进行训练，并使用 PEFT 进行微调。
    """
    # 设置全局种子以确保实验可复现
    set_seed(cfg.seed)

    # 设置训练日志系统
    log, hydra_callback = setup_training_logging(__name__)

    log.info("--- 开始阶段 1：Select-MoE 预训练 ---")
    log.info(f"使用全局种子: {cfg.seed}")

    # 获取分布式训练信息
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    log.info(f"分布式训练信息: local_rank={local_rank}, world_size={world_size}")

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
        full_sequence_prediction=getattr(cfg.training.quality_loss_params, "full_sequence_prediction", False),
        mask_special_tokens=getattr(cfg.training.quality_loss_params, "mask_special_tokens", False),
    )

    # 6. 输出数据统计信息
    # get_data_statistics(tokenized_dataset)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    # 7. 验证并计算批次大小配置
    per_device_batch_size = cfg.training.per_device_batch_size
    gradient_accumulation_steps = validate_batch_size_configuration(cfg.training.batch_size, per_device_batch_size, world_size, log)

    # 8. 配置训练参数
    # LoRA模式：保存中间权重；全秩微调和routing_only：不保存中间权重
    save_strategy = "epoch" if cfg.training.peft_mode == "lora" else "no"

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        gradient_checkpointing=True,
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
            # 保存分词器
            tokenizer.save_pretrained(cfg.output_dir)
            log.info(f"分词器已保存到 {cfg.output_dir}")
            log.info(f"正在将最终的 PEFT 适配模型保存到 {cfg.output_dir}")
            trainer.save_model(cfg.output_dir)
        elif cfg.training.peft_mode == "full_rank":
            log.info(f"正在将路由全秩微调权重保存到 {cfg.output_dir}")
            full_rank_weights_path = os.path.join(cfg.output_dir, "full_rank_weights.pt")
            save_full_rank_weights(model, ROUTING_PATTERNS, full_rank_weights_path, mode="parameter", tokenizer=tokenizer)

    log.info("--- 阶段 1：预训练完成 ---")


if __name__ == "__main__":
    # 这部分用于独立执行。
    # 这里需要一个完整的配置对象。
    pass
