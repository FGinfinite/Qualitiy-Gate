# scripts/verify_changes.py
import os
import sys

import torch
from omegaconf import OmegaConf

# 将项目根目录添加到 Python 路径中，以解决模块导入问题
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stages.pretrain import get_model_and_tokenizer


def verify_changes():
    """
    该脚本旨在通过执行单次前向传播来验证“前向挂钩”和“约束损失”的实现。
    它会加载预训练阶段的配置和模型，处理一个样本输入，
    并打印出详细的损失分解信息，以证明约束损失被正确计算并添加。
    """
    print("--- 开始验证脚本 ---")

    # 1. 加载配置
    print("正在加载配置: configs/stage_1_pretrain.yaml")
    cfg = OmegaConf.load("configs/stage_1_pretrain.yaml")

    # 2. 加载模型和分词器
    print("正在使用 get_model_and_tokenizer 加载模型和分词器...")
    # 注意：我们将模型移动到 CPU 以进行简单的、非 GPU 的验证
    model, tokenizer = get_model_and_tokenizer(cfg)
    # 确保模型在 CPU 上也使用 bfloat16，以匹配其原始 dtype
    model.to(torch.bfloat16).to("cpu")
    print("模型和分词器加载成功。")

    # 3. 创建输入样本
    input_text = "Hello, this is a test sentence to check the model's behavior."
    print(f"\n准备输入样本:\n'{input_text}'")
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    # 将输入移动到 CPU 并转换为 bfloat16
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()  # Causal LM 的典型做法
    print("输入样本已处理。")

    # 4. 执行单次前向传播
    print("\n正在执行单次前向传播...")
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        outputs = model(**inputs)
    print("前向传播完成。")

    # 5. 验证输出
    # 在 TrashCanMoEForCausalLM.forward 方法中添加的打印语句将在这里显示
    print("\n--- 最终输出 ---")
    print(f"从模型输出中获取的最终损失: {outputs.loss.item():.4f}")
    print("\n脚本执行成功。请检查上面的“损失计算验证”部分的输出，")
    print("以确认原始损失和约束损失是否被正确计算。")
    print("\n--- 验证脚本结束 ---")


if __name__ == "__main__":
    verify_changes()
