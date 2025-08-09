from typing import Tuple

import torch
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data import encode_data, get_data_statistics, load_selected_data
from src.utils.logging_utils import setup_training_logging

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
# 核心数据加载和模型初始化
# ---------------------------------------------------------------------------


def get_model_and_tokenizer(
    cfg: DictConfig,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载Llama-2-7B模型和分词器。
    """
    # 加载模型
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **model_kwargs)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


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

    Raises:
        SystemExit: 当批次大小配置无效时退出
    """
    effective_batch_size = per_device_batch_size * world_size

    if total_batch_size % effective_batch_size != 0:
        log.error(f"批次大小配置错误:")
        log.error(f"  总批次大小: {total_batch_size}")
        log.error(f"  每设备批次大小: {per_device_batch_size}")
        log.error(f"  世界大小: {world_size}")
        log.error(f"  有效批次大小: {effective_batch_size}")
        log.error(f"总批次大小 ({total_batch_size}) 必须能被有效批次大小 ({effective_batch_size}) 整除")
        log.error("请调整 training.batch_size 或 training.per_device_batch_size 配置")
        raise SystemExit(1)

    gradient_accumulation_steps = total_batch_size // effective_batch_size

    log.info(f"批次大小配置验证通过:")
    log.info(f"  总批次大小: {total_batch_size}")
    log.info(f"  每设备批次大小: {per_device_batch_size}")
    log.info(f"  世界大小: {world_size}")
    log.info(f"  有效批次大小: {effective_batch_size}")
    log.info(f"  梯度累积步数: {gradient_accumulation_steps}")

    return gradient_accumulation_steps


# ---------------------------------------------------------------------------
# 主要微调阶段
# ---------------------------------------------------------------------------


def finetune(cfg: DictConfig) -> None:
    """
    使用Llama-2-7B模型进行LoRA微调。
    """
    # 设置训练日志系统
    log, hydra_callback = setup_training_logging(__name__)

    log.info("--- 开始阶段 3：Llama-2-7B LoRA 微调 ---")

    set_seed(cfg.training.seed)

    # 获取分布式训练信息
    import os

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    log.info(f"分布式训练信息: local_rank={local_rank}, world_size={world_size}")
    log.info(f"CUDA设备数量: {torch.cuda.device_count()}")
    log.info(f"当前CUDA设备: {torch.cuda.current_device()}")

    if cfg.training.gpu_grab.grab and local_rank == 0:
        # 只在主进程抢占GPU
        log.info("--- 正在抢占GPU显存 ---")
        # 创建一个简化的GPU抢占逻辑，不依赖accelerator
        torch.cuda.empty_cache()
        log.info("--- GPU显存清理完成 ---")

    # 2. 加载 Llama-2-7B 模型和分词器
    log.info("正在加载和配置 Llama-2-7B 模型...")
    model, tokenizer = get_model_and_tokenizer(cfg)
    log.info("模型加载完成。")

    # 3. 配置LoRA微调
    log.info("正在配置LoRA微调...")
    peft_config = get_peft_config(cfg)
    model = get_peft_model(model, peft_config)
    log.info("PEFT 模型已创建。可训练参数：")
    model.print_trainable_parameters()

    # 确保 `use_cache` 已禁用以保证训练兼容性
    model.config.use_cache = False

    # 4. 加载并准备选择后的数据集
    log.info("正在加载选择后的数据集...")
    dataset = load_selected_data(cfg.dataset.data_path)

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

    # 7. 验证并计算批次大小配置
    per_device_batch_size = cfg.training.per_device_batch_size
    gradient_accumulation_steps = validate_batch_size_configuration(
        cfg.training.batch_size, per_device_batch_size, world_size, log
    )

    # 8. 数据整理器
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    # 9. 配置训练参数
    # 基本训练参数
    training_args_dict = {
        "output_dir": cfg.output_dir,
        "num_train_epochs": cfg.training.epochs,
        "per_device_train_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": cfg.training.learning_rate,
        "lr_scheduler_type": cfg.training.scheduler,
        "warmup_ratio": cfg.training.warmup_ratio,
        "logging_dir": f"{cfg.output_dir}/logs",
        "logging_steps": 10,
        "save_strategy": "epoch",
        "report_to": "none",
        "bf16": True,
        "dataloader_drop_last": True,
        "remove_unused_columns": False,
        # 提高稳定性的设置
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": True,
        # 同步设置
        "ddp_timeout": 1800,  # 30分钟超时
    }

    # 根据设备数决定是否使用FSDP
    if world_size > 1:
        log.info(f"检测到多GPU环境 (world_size={world_size})，启用FSDP配置")
        training_args_dict.update({
            "fsdp": "full_shard auto_wrap",
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
        })
    else:
        log.info(f"检测到单GPU环境 (world_size={world_size})，使用常规训练模式（无FSDP）")

    training_args = TrainingArguments(**training_args_dict)

    # 10. 初始化并运行训练器
    # 清理显存
    if cfg.training.gpu_grab.grab:
        torch.cuda.empty_cache()
        if local_rank == 0:
            log.info("已清理GPU显存以准备训练...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[hydra_callback],  # 使用设置好的日志回调
    )

    log.info("正在使用 Hugging Face Trainer 开始LoRA微调...")
    try:
        trainer.train()
        log.info("LoRA微调完成。")
    except Exception as e:
        log.error(f"训练过程中出现错误: {e}")
        # 尝试保存检查点
        try:
            trainer.save_model(f"{cfg.output_dir}/error_checkpoint")
            log.info("已保存错误检查点")
        except Exception as error_save_error:
            log.error(f"无法保存错误检查点: {error_save_error}")
        raise

    # 11. 保存最终模型
    if local_rank == 0:
        log.info(f"正在将最终的 LoRA 适配器保存到 {cfg.output_dir}")
        trainer.save_model(cfg.output_dir)

    log.info("--- 阶段 3：LoRA微调完成 ---")


# 为了兼容性，保留原来的train函数名
def train(cfg: DictConfig) -> None:
    """
    兼容性函数，调用finetune函数
    """
    finetune(cfg)


if __name__ == "__main__":
    # 这部分用于独立执行。
    # 这里需要一个完整的配置对象。
    pass
