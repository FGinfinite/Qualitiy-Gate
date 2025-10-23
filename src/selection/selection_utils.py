#!/usr/bin/env python3
"""
数据选择公共工具模块

提供数据加载、准备和配置保存等公共函数，供 continue_selection.py 和 batch_selection.py 复用。
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import yaml


def load_router_data(router_data_path: str) -> dict:
    """
    加载路由数据

    Args:
        router_data_path: router_data文件路径

    Returns:
        router数据字典
    """
    router_data = torch.load(router_data_path, map_location="cpu")
    return router_data


def load_all_router_data(router_data_dir: str) -> Dict[str, Dict]:
    """
    加载所有数据集的router_data文件

    Args:
        router_data_dir: router_data目录路径

    Returns:
        所有数据集的router数据 {dataset_name: router_data}
    """
    log = logging.getLogger(__name__)
    all_router_data = {}

    if not os.path.exists(router_data_dir):
        raise FileNotFoundError(f"Router data目录不存在: {router_data_dir}")

    # 查找所有的router_data文件
    for filename in os.listdir(router_data_dir):
        if filename.endswith("_router_data.pt"):
            dataset_name = filename.replace("_router_data.pt", "")
            file_path = os.path.join(router_data_dir, filename)

            log.debug(f"加载数据集 '{dataset_name}' 的router data: {file_path}")
            router_data = load_router_data(file_path)
            all_router_data[dataset_name] = router_data

    if not all_router_data:
        raise ValueError(f"在目录 {router_data_dir} 中未找到任何router_data文件")

    log.debug(f"成功加载 {len(all_router_data)} 个数据集的router数据")
    return all_router_data


def load_original_dataset_mapping(router_data_dir: str, data_dir: Optional[str] = None) -> Dict[str, Dict[str, any]]:
    """
    加载原始数据集的消息映射

    Args:
        router_data_dir: router_data文件目录
        data_dir: 数据目录路径（可选）

    Returns:
        数据集映射字典 {dataset_name: {sample_id: messages}}
    """
    log = logging.getLogger(__name__)

    # 获取router_data中的数据集名称
    dataset_names = []
    for filename in os.listdir(router_data_dir):
        if filename.endswith("_router_data.pt"):
            dataset_name = filename.replace("_router_data.pt", "")
            dataset_names.append(dataset_name)

    if not dataset_names:
        log.warning("未找到任何router_data文件")
        return {}

    log.debug(f"需要加载的数据集: {dataset_names}")

    # 根据数据集名称确定加载策略
    from src.data.dataset_loader import load_hf_datasets, load_local_datasets

    dataset_mapping = {}

    for dataset_name in dataset_names:
        log.debug(f"处理数据集: {dataset_name}")

        # 获取数据集配置
        dataset_config = _get_dataset_config(dataset_name)

        if not dataset_config:
            log.warning(f"未知数据集: {dataset_name}，请在get_dataset_config()中添加配置")
            dataset_mapping[dataset_name] = {}
            continue

        source_type = dataset_config["source_type"]
        config = dataset_config["config"]

        try:
            if source_type == "local":
                # 加载本地数据集
                effective_data_dir = data_dir or config["data_dir"]

                # 推断项目根目录
                if not os.path.isabs(effective_data_dir):
                    current_dir = router_data_dir
                    project_root = None
                    for _ in range(10):
                        current_dir = os.path.dirname(current_dir)
                        if os.path.exists(os.path.join(current_dir, effective_data_dir)):
                            project_root = current_dir
                            break

                    if project_root:
                        effective_data_dir = os.path.join(project_root, effective_data_dir)
                    else:
                        effective_data_dir = os.path.join(os.getcwd(), effective_data_dir)

                log.debug(f"从本地加载数据集 '{dataset_name}': {effective_data_dir}")
                combined_dataset = load_local_datasets(data_dir=effective_data_dir, dataset_names=config["dataset_names"], sample_percentage=1.0, seed=0)

            elif source_type == "hf":
                # 加载HuggingFace数据集
                log.debug(f"从HuggingFace加载数据集 '{dataset_name}'")
                combined_dataset = load_hf_datasets(hf_config={"datasets": config["datasets"]}, sample_percentage=1.0, seed=0)
            else:
                log.error(f"不支持的数据源类型: {source_type}")
                dataset_mapping[dataset_name] = {}
                continue

            # 组织数据映射
            single_dataset_mapping = {}
            for item in combined_dataset:
                item_dataset = item.get("dataset")
                item_id = item.get("id")
                messages = item.get("messages", [])

                if item_dataset and item_id and item_dataset == dataset_name:
                    single_dataset_mapping[item_id] = messages

            dataset_mapping[dataset_name] = single_dataset_mapping
            log.debug(f"数据集 '{dataset_name}' 包含 {len(single_dataset_mapping)} 个样本映射")

        except Exception as e:
            log.error(f"加载数据集 '{dataset_name}' 失败: {e}")
            dataset_mapping[dataset_name] = {}

    return dataset_mapping


def prepare_selection_data(
    all_router_data: Dict[str, Dict], dataset_mapping: Dict[str, Dict[str, any]]
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str], List[list], List[str]]:
    """
    准备数据选择所需的所有数据

    Args:
        all_router_data: 所有数据集的router数据
        dataset_mapping: 数据集消息映射

    Returns:
        (all_quality_gates, all_perplexities, all_sample_ids, all_messages, all_dataset_names)
    """
    log = logging.getLogger(__name__)

    all_quality_gates = []
    all_perplexities = []
    all_sample_ids = []
    all_messages = []
    all_dataset_names = []

    for dataset_name, router_data in all_router_data.items():
        quality_gates = router_data["quality_gates"]  # [N, L, max_seq_len]
        perplexities = router_data["perplexities"]  # [N, max_seq_len]
        sample_ids = router_data["sample_ids"]
        num_samples = router_data["num_samples"]

        log.debug(f"处理数据集 '{dataset_name}': {num_samples} 个样本")

        # 获取该数据集的消息映射
        messages_mapping = dataset_mapping.get(dataset_name, {})

        for i in range(num_samples):
            # 获取质量门控和困惑度（去除padding）
            qg = quality_gates[i]  # [L, max_seq_len]
            ppl = perplexities[i]  # [max_seq_len]

            # 找到有效长度（困惑度非零的部分）
            valid_mask = ppl > 0
            valid_length = valid_mask.sum().item()

            if valid_length > 0:
                qg_valid = qg[:, :valid_length]  # [L, T]
                ppl_valid = ppl[:valid_length]  # [T]
            else:
                # 至少保留一个token
                qg_valid = qg[:, :1]
                ppl_valid = ppl[:1]

            all_quality_gates.append(qg_valid)
            all_perplexities.append(ppl_valid)

            # 获取原始messages
            sample_id = sample_ids[i]
            messages = messages_mapping.get(sample_id, [])

            if not messages:
                log.warning(f"样本 {sample_id} 未找到对应的messages")

            all_sample_ids.append(sample_id)
            all_messages.append(messages)
            all_dataset_names.append(dataset_name)

    log.info(f"共准备 {len(all_quality_gates)} 个样本")

    return all_quality_gates, all_perplexities, all_sample_ids, all_messages, all_dataset_names


def save_selection_config(
    output_dir: str,
    script_name: str,
    selection_percentage: float,
    alpha: float,
    eps: float,
    tau: float,
    router_data_dir: str,
    selected_data_path: str,
    data_dir: Optional[str],
    start_time: datetime,
    end_time: datetime,
):
    """
    保存数据选择配置到selection_config.yaml

    Args:
        output_dir: 输出目录
        script_name: 脚本名称
        selection_percentage: 选择比例
        alpha: 困惑度权重参数
        eps: 数值稳定性参数
        tau: 层权重温度参数
        router_data_dir: router_data目录路径
        selected_data_path: 选择数据输出路径
        data_dir: 数据目录路径
        start_time: 开始时间
        end_time: 结束时间
    """
    log = logging.getLogger(__name__)

    # 构建配置字典
    config_dict = {
        "selection_metadata": {
            "script_name": script_name,
            "execution_time": {"start": start_time.isoformat(), "end": end_time.isoformat(), "duration_seconds": (end_time - start_time).total_seconds()},
        },
        "selection_parameters": {
            "selection_percentage": float(selection_percentage),
            "method": "quality_based",
            "alpha": float(alpha),
            "eps": float(eps),
            "tau": float(tau),
        },
        "paths": {
            "router_data_dir": router_data_dir,
            "selected_data_path": selected_data_path,
            "data_dir": data_dir or "auto-inferred",
        },
    }

    # 保存配置文件
    config_path = os.path.join(output_dir, "selection_config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    log.debug(f"选择配置已保存到: {config_path}")


def generate_output_path(router_data_dir: str) -> str:
    """
    根据router_data_dir自动生成output_path
    将输出文件放在与router_data_dir同一父目录下

    Args:
        router_data_dir: router_data目录路径

    Returns:
        生成的输出文件路径
    """
    # 获取父目录
    parent_dir = os.path.dirname(router_data_dir)
    # 生成输出文件路径
    output_path = os.path.join(parent_dir, "selected_data.jsonl")
    return output_path


def _get_dataset_config(dataset_name: str) -> dict:
    """
    数据集配置工厂函数

    返回数据集的配置信息，包括数据源类型和加载参数

    Args:
        dataset_name: 数据集名称

    Returns:
        数据集配置字典，包含:
        - source_type: "local" 或 "hf"
        - config: 对应的加载配置
    """
    # 数据集名称到配置的映射表
    # 添加新数据集时，只需在这里添加相应的映射即可
    dataset_configs = {
        # 本地数据集
        "cot": {"source_type": "local", "config": {"data_dir": "dataset/train/processed", "dataset_names": ["cot"]}},
        "dolly": {"source_type": "local", "config": {"data_dir": "dataset/train/processed", "dataset_names": ["dolly"]}},
        "flan_v2": {"source_type": "local", "config": {"data_dir": "dataset/train/processed", "dataset_names": ["flan_v2"]}},
        "oasst1": {"source_type": "local", "config": {"data_dir": "dataset/train/processed", "dataset_names": ["oasst1"]}},
        # HuggingFace数据集
        "openhermes": {
            "source_type": "hf",
            "config": {"datasets": [{"name": "teknium/OpenHermes-2.5", "dataset_name": "openhermes", "subset": None, "split": "train"}]},
        },
        "gsm8k": {
            "source_type": "hf",
            "config": {"datasets": [{"name": "openai/gsm8k", "dataset_name": "gsm8k", "subset": "main", "split": "train"}]},
        },
        # HENDRYCKS_MATH 数据集的所有子集
        "algebra": {
            "source_type": "hf",
            "config": {"datasets": [{"name": "EleutherAI/hendrycks_math", "dataset_name": "algebra", "subset": "algebra", "split": "train"}]},
        },
        "counting_and_probability": {
            "source_type": "hf",
            "config": {
                "datasets": [
                    {"name": "EleutherAI/hendrycks_math", "dataset_name": "counting_and_probability", "subset": "counting_and_probability", "split": "train"}
                ]
            },
        },
        "geometry": {
            "source_type": "hf",
            "config": {"datasets": [{"name": "EleutherAI/hendrycks_math", "dataset_name": "geometry", "subset": "geometry", "split": "train"}]},
        },
        "intermediate_algebra": {
            "source_type": "hf",
            "config": {
                "datasets": [{"name": "EleutherAI/hendrycks_math", "dataset_name": "intermediate_algebra", "subset": "intermediate_algebra", "split": "train"}]
            },
        },
        "number_theory": {
            "source_type": "hf",
            "config": {"datasets": [{"name": "EleutherAI/hendrycks_math", "dataset_name": "number_theory", "subset": "number_theory", "split": "train"}]},
        },
        "prealgebra": {
            "source_type": "hf",
            "config": {"datasets": [{"name": "EleutherAI/hendrycks_math", "dataset_name": "prealgebra", "subset": "prealgebra", "split": "train"}]},
        },
        "precalculus": {
            "source_type": "hf",
            "config": {"datasets": [{"name": "EleutherAI/hendrycks_math", "dataset_name": "precalculus", "subset": "precalculus", "split": "train"}]},
        },
        # 未来添加新的HF数据集示例:
        # "alpaca": {
        #     "source_type": "hf",
        #     "config": {
        #         "datasets": [{
        #             "name": "tatsu-lab/alpaca",
        #             "dataset_name": "alpaca",
        #             "subset": None,
        #             "split": "train"
        #         }]
        #     }
        # }
    }

    return dataset_configs.get(dataset_name, {})
