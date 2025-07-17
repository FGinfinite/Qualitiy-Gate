from typing import Tuple

import torch
from accelerate import Accelerator
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
from utils.tools import grab_gpu

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


def calculate_per_device_batch_size(total_batch_size: int, num_processes: int) -> int:
    """
    根据总批次大小和进程数计算每个设备的批次大小
    """
    per_device_batch_size = total_batch_size // num_processes
    if total_batch_size % num_processes != 0:
        print(f"警告: 总批次大小 {total_batch_size} 不能被进程数 {num_processes} 整除")
        print(f"使用每设备批次大小: {per_device_batch_size}")
    return per_device_batch_size


# ---------------------------------------------------------------------------
# 主要微调阶段
# ---------------------------------------------------------------------------


def finetune(cfg: DictConfig) -> None:
    """
    使用Llama-2-7B模型进行LoRA微调。
    """

    print("--- 开始阶段 3：Llama-2-7B LoRA 微调 ---")

    accelerator = Accelerator()
    set_seed(cfg.training.seed)

    if cfg.training.gpu_grab.grab:
        # 1. 抢占GPU并持有占位符
        print("--- 正在抢占GPU显存 ---")
        placeholders = grab_gpu(
            memory_need=cfg.training.gpu_grab.memory_need_gb,
            accelerator=accelerator,
            over_grab=cfg.training.gpu_grab.over_grab,
        )
        print("--- GPU显存抢占完成 ---")

    # 2. 加载 Llama-2-7B 模型和分词器
    print("正在加载和配置 Llama-2-7B 模型...")
    model, tokenizer = get_model_and_tokenizer(cfg)
    print("模型加载完成。")

    # 3. 配置LoRA微调
    print("正在配置LoRA微调...")
    peft_config = get_peft_config(cfg)
    model = get_peft_model(model, peft_config)
    print("PEFT 模型已创建。可训练参数：")
    model.print_trainable_parameters()

    # 确保 `use_cache` 已禁用以保证训练兼容性
    model.config.use_cache = False

    # 4. 加载并准备选择后的数据集
    print("正在加载选择后的数据集...")
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

    # 7. 计算每设备批次大小
    num_processes = accelerator.num_processes
    per_device_batch_size = calculate_per_device_batch_size(
        cfg.training.batch_size, num_processes
    )
    print(f"总批次大小: {cfg.training.batch_size}")
    print(f"进程数: {num_processes}")
    print(f"每设备批次大小: {per_device_batch_size}")

    # 8. 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding="longest"
    )

    # 9. 配置训练参数
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.scheduler,
        warmup_ratio=cfg.training.warmup_ratio,
        logging_dir=f"{cfg.output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        bf16=True,  # 使用bf16混合精度
        dataloader_drop_last=True,
        remove_unused_columns=False,
    )

    # 10. 初始化并运行训练器
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

    print("正在使用 Hugging Face Trainer 开始LoRA微调...")
    # 使用 try...finally 确保训练过程的稳健性
    try:
        trainer.train()
    finally:
        # 可以在此处添加清理代码（如果需要）
        pass
    print("LoRA微调完成。")

    # 11. 保存最终模型
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"正在将最终的 LoRA 适配器保存到 {cfg.output_dir}")
        trainer.save_model(cfg.output_dir)

    print("\n--- 阶段 3：LoRA微调完成 ---")


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
