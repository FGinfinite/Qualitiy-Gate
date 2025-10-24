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

from src.data import encode_data, get_data_statistics, load_and_prepare_dataset, load_selected_data
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
    加载模型和分词器。
    """
    # 加载模型
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(cfg.training.model.name, **model_kwargs)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(cfg.training.model.name)
    if tokenizer.pad_token is None:
        # 添加特殊的填充token，而不是使用eos_token
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

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
# 主要微调阶段
# ---------------------------------------------------------------------------


def finetune(cfg: DictConfig) -> None:
    """
    进行LoRA微调。
    """
    # 设置训练日志系统
    log, hydra_callback = setup_training_logging(__name__)

    log.info(f"--- 开始阶段 3：{cfg.training.model.name} LoRA 微调 ---")
    log.info(f"使用全局种子: {cfg.seed}")

    set_seed(cfg.seed)

    # 获取分布式训练信息
    import os

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    log.info(f"分布式训练信息: local_rank={local_rank}, world_size={world_size}")
    log.info(f"CUDA设备数量: {torch.cuda.device_count()}")
    log.info(f"当前CUDA设备: {torch.cuda.current_device()}")

    if cfg.gpu_grab.grab and local_rank == 0:
        # 只在主进程抢占GPU
        log.info("--- 正在抢占GPU显存 ---")
        # 创建一个简化的GPU抢占逻辑，不依赖accelerator
        torch.cuda.empty_cache()
        log.info("--- GPU显存清理完成 ---")

    # 2. 加载模型和分词器
    log.info(f"正在加载和配置 {cfg.training.model.name} 模型...")
    model, tokenizer = get_model_and_tokenizer(cfg)
    log.info("模型加载完成。")

    # 3. 配置LoRA微调
    log.info("正在配置LoRA微调...")
    peft_config = get_peft_config(cfg)
    model = get_peft_model(model, peft_config)
    log.info("PEFT 模型已创建。可训练参数：")
    model.print_trainable_parameters()

    # 4. 调整embedding大小并处理梯度（匹配LESS项目的逻辑）
    from peft import PeftModel

    # 检查是否需要调整embedding大小
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        log.info(f"调整embedding大小: {embedding_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        # 如果是PEFT模型且调整了embedding，需要禁用embedding的梯度
        if isinstance(model, PeftModel):
            model.get_input_embeddings().weight.requires_grad = False
            model.get_output_embeddings().weight.requires_grad = False
            log.info("已禁用embedding层的梯度更新")

    # 启用输入梯度（用于gradient checkpointing）
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    log.info("已启用输入梯度计算")

    # 确保 `use_cache` 已禁用以保证训练兼容性
    model.config.use_cache = False

    # 5. 加载并准备数据集
    if cfg.dataset.mode == "full":
        log.info("正在加载全量训练数据集...")
        dataset = load_and_prepare_dataset(cfg)
    else:
        log.info("正在加载选择后的数据集...")
        dataset = load_selected_data(cfg.dataset.data_path)
        # 对选择数据也应用shuffle
        if getattr(cfg.dataset, "shuffle", True):
            dataset = dataset.shuffle(seed=cfg.seed)
            log.info(f"已打乱数据集，使用种子: {cfg.seed}")

    # 6. 编码数据集
    tokenized_dataset = encode_data(
        dataset,
        tokenizer,
        max_seq_length=cfg.dataset.max_sequence_length,
        processing_num_workers=getattr(cfg.dataset, "processing_num_workers", 10),
        overwrite_cache=getattr(cfg.dataset, "overwrite_cache", False),
    )

    # 6. 输出数据统计信息
    # get_data_statistics(tokenized_dataset)

    # 7. 验证并计算批次大小配置
    per_device_batch_size = cfg.training.per_device_batch_size
    gradient_accumulation_steps = validate_batch_size_configuration(cfg.training.batch_size, per_device_batch_size, world_size, log)

    # 8. 数据整理器
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    # 9. 配置训练参数
    # 确定PEFT检查点保存路径
    peft_output_dir = f"{cfg.output_dir}/{cfg.checkpoint.peft_dir}"
    log.info(f"PEFT适配器将保存到: {peft_output_dir}")

    # 基本训练参数
    training_args_dict = {
        "output_dir": peft_output_dir,  # PEFT检查点保存到PEFT子目录
        "num_train_epochs": cfg.training.epochs,
        "per_device_train_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": cfg.training.learning_rate,
        "lr_scheduler_type": cfg.training.scheduler,
        "warmup_ratio": cfg.training.warmup_ratio,
        "weight_decay": cfg.training.weight_decay,
        "logging_dir": f"{cfg.output_dir}/logs",
        "logging_steps": 10,
        "save_strategy": "epoch" if cfg.checkpoint.save_intermediate else "no",  # 根据配置决定是否保存中间检查点
        "report_to": "none",
        "bf16": True,
        "tf32": False,  # 禁用TensorFloat-32以匹配LESS配置
        "dataloader_drop_last": True,
        "remove_unused_columns": False,
        # 提高稳定性的设置
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": True,
        # 同步设置
        "ddp_timeout": 1800,  # 30分钟超时
    }

    FSDP_CONFIG = {
        "meta-llama/Llama-2-7b-hf": {
            "fsdp": "full_shard auto_wrap",
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
        },
        "Qwen/Qwen3-1.7B": {
            "fsdp": "full_shard auto_wrap",
            "fsdp_transformer_layer_cls_to_wrap": "Qwen3DecoderLayer",
        },
        "Qwen/Qwen3-1.7B-Base": {
            "fsdp": "full_shard auto_wrap",
            "fsdp_transformer_layer_cls_to_wrap": "Qwen3DecoderLayer",
        },
        "Qwen/Qwen2.5-1.5B": {
            "fsdp": "full_shard auto_wrap",
            "fsdp_transformer_layer_cls_to_wrap": "Qwen2DecoderLayer",
        },
    }

    # 根据设备数决定是否使用FSDP
    if world_size > 1:
        log.info(f"检测到多GPU环境 (world_size={world_size})，启用FSDP配置")
        training_args_dict.update(FSDP_CONFIG[cfg.training.model.name])
    else:
        log.info(f"检测到单GPU环境 (world_size={world_size})，使用常规训练模式（无FSDP）")

    training_args = TrainingArguments(**training_args_dict)

    # 10. 初始化并运行训练器
    # 清理显存
    if cfg.gpu_grab.grab:
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
            trainer.save_model(f"{peft_output_dir}/error_checkpoint")
            log.info("已保存错误检查点")
        except Exception as error_save_error:
            log.error(f"无法保存错误检查点: {error_save_error}")
        raise

    # 11. 保存最终的PEFT适配器
    log.info(f"正在将最终的 LoRA 适配器保存到 {peft_output_dir}")
    trainer.save_model(peft_output_dir)
    log.info("PEFT适配器保存完成。")

    # 12. 如果配置要求，保存合并后的完整模型
    if cfg.checkpoint.save_merged_model:
        # 在FSDP模式下，需要等待所有进程完成PEFT适配器的保存
        if world_size > 1:
            import torch.distributed as dist

            dist.barrier()
            log.info("所有进程已完成PEFT适配器保存，准备合并模型...")

        # 只在主进程（rank 0）执行合并和保存
        if local_rank == 0:
            log.info("正在合并PEFT适配器到基座模型...")
            log.info("为确保在FSDP模式下正确工作，将重新加载基座模型和PEFT适配器...")

            # 清理当前模型以释放内存
            del model
            del trainer
            torch.cuda.empty_cache()

            # 重新加载基座模型（在CPU或单GPU上，避免FSDP分片）
            log.info(f"重新加载基座模型: {cfg.training.model.name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                cfg.training.model.name,
                torch_dtype=torch.bfloat16,
                device_map="auto",  # 自动分配到可用设备
                low_cpu_mem_usage=True,
            )

            # 加载刚保存的PEFT适配器
            log.info(f"加载PEFT适配器: {peft_output_dir}")
            from peft import PeftModel

            peft_model = PeftModel.from_pretrained(base_model, peft_output_dir)

            # 合并PEFT权重到基座模型
            log.info("正在合并PEFT权重...")
            merged_model = peft_model.merge_and_unload()

            # 保存合并后的完整模型到输出目录根路径
            merged_model_path = cfg.output_dir
            log.info(f"正在将合并后的完整模型保存到 {merged_model_path}")

            merged_model.save_pretrained(merged_model_path, safe_serialization=True, max_shard_size="5GB")

            # 同时保存tokenizer
            tokenizer.save_pretrained(merged_model_path)

            # 清理内存
            del merged_model
            del peft_model
            del base_model
            torch.cuda.empty_cache()

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
