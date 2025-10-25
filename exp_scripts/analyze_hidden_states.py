#!/usr/bin/env python3
"""
分析高质量数据和低质量数据的hidden states差异
用于理解为什么随机初始化的Quality Gate能区分数据质量

运行指令：
CUDA_VISIBLE_DEVICES=0 python -m exp_scripts.analyze_hidden_states \
    --model-path converted_models/quality_gate_Qwen3-1.7B-Base \
    --high-quality-dataset openai/gsm8k \
    --low-quality-dataset dataset/train/processed/faker_dataset_24k \
    --num-samples 100
"""

import argparse
import glob
import json
import logging

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from src.models.quality_gate_model import QualityGateForCausalLM, register_quality_gate


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_model_and_tokenizer(model_path: str, device: str):
    """加载模型和分词器"""
    log = logging.getLogger(__name__)

    # 注册Quality Gate模型
    register_quality_gate()

    log.info(f"加载模型: {model_path}")
    model = QualityGateForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    log.info("加载分词器")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def load_and_format_dataset(dataset_path: str, num_samples: int):
    """
    加载数据集并转换为统一的 messages 格式

    支持：
    1. 本地 jsonl 文件（已经是 messages 格式）
    2. HuggingFace 数据集（需要格式转换）
    """
    log = logging.getLogger(__name__)

    # 判断是本地路径还是 HF 数据集
    if "/" in dataset_path and not dataset_path.startswith("dataset/"):
        # HuggingFace 数据集
        log.info(f"从 HuggingFace 加载数据集: {dataset_path}")

        # 尝试加载数据集，如果失败则尝试使用默认config
        try:
            dataset = load_dataset(dataset_path, split="train")
        except ValueError as e:
            if "Config name is missing" in str(e):
                # 提取可用的configs
                log.info("数据集需要指定config，尝试使用 'main' 作为默认config")
                dataset = load_dataset(dataset_path, "main", split="train")
            else:
                raise

        # 检测数据格式并转换
        if "messages" in dataset.column_names:
            # 已经是 messages 格式
            pass
        elif "question" in dataset.column_names and "answer" in dataset.column_names:
            # GSM8K 格式
            log.info("检测到 GSM8K 格式，转换为 messages 格式")

            def convert_to_messages(example):
                return {
                    "messages": [
                        {"role": "user", "content": example["question"]},
                        {"role": "assistant", "content": example["answer"]},
                    ]
                }

            dataset = dataset.map(convert_to_messages)
        else:
            raise ValueError(f"不支持的数据格式，字段: {dataset.column_names}")
    else:
        # 本地 jsonl 文件
        log.info(f"从本地加载数据集: {dataset_path}")
        jsonl_files = glob.glob(f"{dataset_path}/*.jsonl") + glob.glob(f"{dataset_path}/*.json")
        if not jsonl_files:
            raise ValueError(f"在 {dataset_path} 中找不到 jsonl 文件")
        dataset = load_dataset("json", data_files=jsonl_files)["train"]

    # 采样
    if num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    log.info(f"加载了 {len(dataset)} 个样本")
    return dataset


def collect_hidden_states(model, tokenizer, dataset, num_samples: int, device: str):
    """
    收集hidden states及相关统计量

    Returns:
        stats: dict with keys:
            - hidden_states_before_gate: List of [seq_len, hidden_size]
            - quality_scores: List of [seq_len, 1]
            - token_ids: List of token sequences
    """
    log = logging.getLogger(__name__)
    stats = {
        "hidden_states_before_gate": [],
        "quality_scores_per_layer": [],  # [num_samples, num_layers, seq_len]
        "token_ids": [],
        "attention_patterns": [],
    }

    log.info(f"开始收集 {num_samples} 个样本的hidden states...")

    # 注册hook来捕获hidden states
    hidden_states_cache = []

    def hook_fn(module, inputs, output):
        # 在quality_gate之前捕获hidden states
        # inputs是一个tuple，第一个元素是hidden_states
        hidden_states_cache.append(inputs[0].detach().cpu())

    # 为第一层的quality_gate注册hook
    hook = model.model.layers[0].quality_gate.register_forward_hook(hook_fn)

    with torch.no_grad():
        for sample in tqdm(dataset):
            messages = sample["messages"]

            # 转换为文本
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=2048,
            ).to(device)

            # 前向传播
            hidden_states_cache.clear()
            outputs = model(**inputs, output_router_logits=True, output_hidden_states=True)

            # 收集质量分数
            quality_scores = [rl.detach().cpu() for rl in outputs.router_logits]  # List of [1, seq_len, 1]
            quality_scores = torch.stack([qs.squeeze(0) for qs in quality_scores], dim=0)  # [num_layers, seq_len, 1]

            # 保存统计量
            stats["hidden_states_before_gate"].append(hidden_states_cache[0].squeeze(0))  # [seq_len, hidden_size]
            stats["quality_scores_per_layer"].append(quality_scores)  # [num_layers, seq_len, 1]
            stats["token_ids"].append(inputs["input_ids"].cpu().squeeze(0))  # [seq_len]

    hook.remove()
    return stats


def analyze_statistics(high_quality_stats, low_quality_stats):
    """分析高质量和低质量数据的hidden states统计差异"""
    log = logging.getLogger(__name__)

    log.info("\n" + "=" * 80)
    log.info("Hidden States 统计分析")
    log.info("=" * 80)

    # 1. 分析hidden states的范数
    log.info("\n1. Hidden States L2范数分析")
    hq_norms = [h.norm(dim=-1).mean().item() for h in high_quality_stats["hidden_states_before_gate"]]
    lq_norms = [h.norm(dim=-1).mean().item() for h in low_quality_stats["hidden_states_before_gate"]]

    log.info(f"   高质量数据 L2范数: mean={sum(hq_norms) / len(hq_norms):.4f}, std={torch.tensor(hq_norms).std().item():.4f}")
    log.info(f"   低质量数据 L2范数: mean={sum(lq_norms) / len(lq_norms):.4f}, std={torch.tensor(lq_norms).std().item():.4f}")
    log.info(f"   差异: {(sum(hq_norms) / len(hq_norms) - sum(lq_norms) / len(lq_norms)):.4f}")

    # 2. 分析hidden states的方差
    log.info("\n2. Hidden States 方差分析")
    hq_vars = [h.var(dim=-1).mean().item() for h in high_quality_stats["hidden_states_before_gate"]]
    lq_vars = [h.var(dim=-1).mean().item() for h in low_quality_stats["hidden_states_before_gate"]]

    log.info(f"   高质量数据方差: mean={sum(hq_vars) / len(hq_vars):.6f}, std={torch.tensor(hq_vars).std().item():.6f}")
    log.info(f"   低质量数据方差: mean={sum(lq_vars) / len(lq_vars):.6f}, std={torch.tensor(lq_vars).std().item():.6f}")

    # 3. 分析hidden states的均值
    log.info("\n3. Hidden States 均值分析")
    hq_means = [h.mean(dim=-1).mean().item() for h in high_quality_stats["hidden_states_before_gate"]]
    lq_means = [h.mean(dim=-1).mean().item() for h in low_quality_stats["hidden_states_before_gate"]]

    log.info(f"   高质量数据均值: mean={sum(hq_means) / len(hq_means):.6f}, std={torch.tensor(hq_means).std().item():.6f}")
    log.info(f"   低质量数据均值: mean={sum(lq_means) / len(lq_means):.6f}, std={torch.tensor(lq_means).std().item():.6f}")

    # 4. 分析质量分数分布
    log.info("\n4. Quality Score 分布分析（未训练模型）")
    hq_scores = torch.cat([torch.sigmoid(qs).mean(dim=0) for qs in high_quality_stats["quality_scores_per_layer"]])
    lq_scores = torch.cat([torch.sigmoid(qs).mean(dim=0) for qs in low_quality_stats["quality_scores_per_layer"]])

    log.info(f"   高质量数据质量分数: mean={hq_scores.mean().item():.4f}, std={hq_scores.std().item():.4f}")
    log.info(f"   低质量数据质量分数: mean={lq_scores.mean().item():.4f}, std={lq_scores.std().item():.4f}")
    log.info(f"   差异: {(hq_scores.mean() - lq_scores.mean()).item():.4f}")

    # 5. 分析维度级别的激活模式
    log.info("\n5. Hidden States 维度激活模式分析")
    # 由于序列长度不同，先对每个样本在时间维度上取平均，再对样本维度取平均
    hq_dim_means_per_sample = [h.mean(dim=0) for h in high_quality_stats["hidden_states_before_gate"]]  # List of [hidden_size]
    lq_dim_means_per_sample = [h.mean(dim=0) for h in low_quality_stats["hidden_states_before_gate"]]  # List of [hidden_size]

    hq_dim_means = torch.stack(hq_dim_means_per_sample).mean(dim=0)  # [hidden_size]
    lq_dim_means = torch.stack(lq_dim_means_per_sample).mean(dim=0)  # [hidden_size]

    dim_diff = (hq_dim_means - lq_dim_means).abs()
    log.info(f"   维度差异L2范数: {dim_diff.norm().item():.6f}")
    log.info(f"   维度差异最大值: {dim_diff.max().item():.6f}")
    log.info(f"   维度差异均值: {dim_diff.mean().item():.6f}")

    # 6. 计算随机线性投影的期望输出差异
    log.info("\n6. 随机线性投影分析")
    # 模拟100个随机投影
    num_projections = 100
    std = 0.02  # 与Quality Gate初始化一致

    projection_diffs = []
    for _ in range(num_projections):
        # 生成随机权重，确保数据类型匹配
        weight = (torch.randn(1, hq_dim_means.size(0)) * std).to(hq_dim_means.dtype)  # [1, hidden_size]

        # 计算投影
        hq_proj = (weight @ hq_dim_means.unsqueeze(-1)).item()
        lq_proj = (weight @ lq_dim_means.unsqueeze(-1)).item()

        projection_diffs.append(abs(hq_proj - lq_proj))

    log.info(f"   随机投影差异均值: {sum(projection_diffs) / len(projection_diffs):.6f}")
    log.info(f"   随机投影差异标准差: {torch.tensor(projection_diffs).std().item():.6f}")
    log.info(f"   随机投影差异最大值: {max(projection_diffs):.6f}")

    log.info("\n" + "=" * 80)

    return {
        "hq_l2_norm": sum(hq_norms) / len(hq_norms),
        "lq_l2_norm": sum(lq_norms) / len(lq_norms),
        "hq_var": sum(hq_vars) / len(hq_vars),
        "lq_var": sum(lq_vars) / len(lq_vars),
        "hq_quality_score": hq_scores.mean().item(),
        "lq_quality_score": lq_scores.mean().item(),
        "dim_diff_norm": dim_diff.norm().item(),
        "random_proj_diff_mean": sum(projection_diffs) / len(projection_diffs),
    }


def main():
    parser = argparse.ArgumentParser(description="分析高质量和低质量数据的hidden states差异")
    parser.add_argument(
        "--model-path",
        type=str,
        default="converted_models/quality_gate_Qwen3-1.7B",
        help="Quality Gate模型路径（未训练）",
    )
    parser.add_argument(
        "--high-quality-dataset",
        type=str,
        default="dataset/train/processed/openhermes",
        help="高质量数据集路径",
    )
    parser.add_argument(
        "--low-quality-dataset",
        type=str,
        default="dataset/train/processed/faker_dataset_24k",
        help="低质量数据集路径（faker）",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="每个数据集采样数量",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="运行设备",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hidden_states_analysis.json",
        help="输出统计结果的文件路径",
    )

    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger(__name__)

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

    # 加载数据集
    log.info("\n=== 加载高质量数据集 ===")
    high_quality_dataset = load_and_format_dataset(args.high_quality_dataset, args.num_samples)

    log.info("\n=== 加载低质量数据集 ===")
    low_quality_dataset = load_and_format_dataset(args.low_quality_dataset, args.num_samples)

    # 收集统计量
    log.info("\n=== 收集高质量数据统计量 ===")
    high_quality_stats = collect_hidden_states(model, tokenizer, high_quality_dataset, args.num_samples, args.device)

    log.info("\n=== 收集低质量数据统计量 ===")
    low_quality_stats = collect_hidden_states(model, tokenizer, low_quality_dataset, args.num_samples, args.device)

    # 分析差异
    analysis_results = analyze_statistics(high_quality_stats, low_quality_stats)

    # 保存结果
    with open(args.output, "w") as f:
        json.dump(analysis_results, f, indent=2)

    log.info(f"\n分析结果已保存到: {args.output}")
    log.info("\n结论：")
    log.info("如果高质量和低质量数据的hidden states在统计上有显著差异，")
    log.info("那么即使是随机初始化的线性投影也可能捕捉到这种差异。")
    log.info("这解释了为什么未训练的Quality Gate能够区分数据质量。")


if __name__ == "__main__":
    main()
