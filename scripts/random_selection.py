#!/usr/bin/env python3
"""
随机数据选择脚本

从指定的数据集中随机选择数据，作为对比的基线。
使用Hydra进行配置管理。
"""

import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Dict, List

import hydra
import yaml
from omegaconf import DictConfig

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.dataset_loader import convert_openhermes_format, load_hf_datasets, load_local_datasets
from src.utils.hydra_resolvers import register_custom_resolvers

# Register custom Hydra resolvers before @hydra.main
register_custom_resolvers()


def setup_logging(cfg: DictConfig):
    """设置日志配置"""
    verbose = cfg.logging.get("verbose", False)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def load_datasets(cfg: DictConfig):
    """加载数据集（返回迭代器，不转换为列表）"""
    log = logging.getLogger(__name__)

    if cfg.dataset.dataset_from == "local":
        log.info("从本地加载数据集...")
        config = cfg.dataset.local
        combined_dataset = load_local_datasets(
            data_dir=config.data_dir, dataset_names=config.dataset_names, sample_percentage=cfg.sample_percentage, seed=cfg.seed
        )
    elif cfg.dataset.dataset_from == "hf":
        log.info("从HuggingFace加载数据集...")
        config = cfg.dataset.hf
        combined_dataset = load_hf_datasets(hf_config={"datasets": config.datasets}, sample_percentage=cfg.sample_percentage, seed=cfg.seed)
    else:
        raise ValueError(f"不支持的数据源类型: {cfg.dataset.dataset_from}")

    # 对于本地数据集，检查并转换格式（HF数据集已在 load_hf_datasets 中转换）
    if cfg.dataset.dataset_from == "local" and len(combined_dataset) > 0:
        sample = combined_dataset[0]

        # 如果包含 conversations 字段但没有 messages 字段，说明是 OpenHermes 格式，需要转换
        if "conversations" in sample and "messages" not in sample:
            log.info("检测到本地 OpenHermes 格式数据，正在转换为标准格式...")

            # 获取数据集名称（如果有）
            dataset_name = sample.get("dataset", "unknown")

            # 获取所有需要移除的原始字段
            columns_to_remove = [col for col in combined_dataset.column_names if col not in ["dataset", "id"]]
            log.info(f"将移除以下原始字段: {columns_to_remove}")

            # 使用 map 进行批量转换，并移除原始字段
            combined_dataset = combined_dataset.map(
                lambda example, idx: convert_openhermes_format(example, dataset_name=example.get("dataset", dataset_name), example_index=idx),
                with_indices=True,
                desc="转换数据集格式",
                load_from_cache_file=True,
                remove_columns=columns_to_remove,  # 移除所有原始字段
            )
            log.info("格式转换完成")
        elif "messages" in sample:
            log.info("数据集已经是标准格式")
        else:
            log.warning("未识别的数据格式，可能导致后续处理失败")

    log.info(f"数据集加载完成，共 {len(combined_dataset)} 个样本")
    return combined_dataset


def streaming_random_selection(dataset_iterator, selection_percentage: float, seed: int) -> List[Dict]:
    """使用 Reservoir Sampling 算法进行流式随机选择"""
    log = logging.getLogger(__name__)

    # 设置随机种子
    random.seed(seed)

    log.info("开始流式随机选择...")
    log.info("第一遍遍历：计算数据集总数和统计分布...")

    # 首先遍历一次获取总数和统计信息
    total_samples = 0
    dataset_stats = {}

    # 将迭代器转换为列表以便多次遍历（但分批处理以节省内存）
    dataset_items = []

    for item in dataset_iterator:
        dataset_items.append(item)
        total_samples += 1

        # 统计数据集分布
        dataset_name = item.get("dataset", "unknown")
        dataset_stats[dataset_name] = dataset_stats.get(dataset_name, 0) + 1

        # 定期报告进度
        if total_samples % 50000 == 0:
            log.info(f"已处理 {total_samples} 个样本...")

    target_count = int(total_samples * selection_percentage)

    log.info(f"数据集总数: {total_samples}")
    log.info("数据集分布:")
    for dataset_name, count in dataset_stats.items():
        log.info(f"  {dataset_name}: {count} 个样本")
    log.info(f"目标选择数量: {target_count} ({selection_percentage * 100:.2f}%)")

    if target_count == 0:
        log.warning("目标选择数量为0，返回空列表")
        return []

    if target_count > total_samples:
        log.warning(f"目标选择数量 ({target_count}) 大于总样本数 ({total_samples})，将选择所有样本")
        return dataset_items

    log.info("第二遍遍历：使用 Reservoir Sampling 进行选择...")

    # 使用 Reservoir Sampling 算法
    selected_data = []

    for i, item in enumerate(dataset_items):
        if len(selected_data) < target_count:
            # 水库未满，直接添加
            selected_data.append(item)
        else:
            # 水库已满，以概率 target_count/i 替换
            j = random.randint(0, i)
            if j < target_count:
                selected_data[j] = item

        # 定期报告进度
        if (i + 1) % 50000 == 0:
            log.info(f"已处理 {i + 1}/{total_samples} 个样本...")

    log.info(f"流式随机选择完成，选择了 {len(selected_data)} 个样本")

    # 统计选择结果中每个数据集的分布
    selected_stats = {}
    for item in selected_data:
        dataset_name = item.get("dataset", "unknown")
        selected_stats[dataset_name] = selected_stats.get(dataset_name, 0) + 1

    log.info("选择结果中的数据集分布:")
    for dataset_name, count in selected_stats.items():
        log.info(f"  {dataset_name}: {count} 个样本")

    return selected_data


def save_selection_config(output_path: str, cfg: DictConfig, start_time: datetime, end_time: datetime):
    """保存随机选择配置到selection_config.yaml"""
    log = logging.getLogger(__name__)

    # 构建配置字典
    config_dict = {
        "selection_metadata": {
            "script_name": "random_selection.py",
            "selection_method": "random",
            "execution_time": {"start": start_time.isoformat(), "end": end_time.isoformat(), "duration_seconds": (end_time - start_time).total_seconds()},
        },
        "selection_parameters": {"selection_percentage": float(cfg.selection_percentage), "seed": cfg.seed, "sample_percentage": float(cfg.sample_percentage)},
        "dataset_configuration": {"dataset_from": cfg.dataset.dataset_from, "dataset_config": dict(cfg.dataset[cfg.dataset.dataset_from])},
        "output_paths": {"selected_data_path": output_path, "output_filename": cfg.output_filename},
    }

    # 保存配置文件
    config_path = os.path.join(os.path.dirname(output_path), "selection_config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    log.info(f"选择配置已保存到: {config_path}")


@hydra.main(config_path="../configs", config_name="random_selection", version_base=None)
def main(cfg: DictConfig) -> None:
    """主函数：执行随机数据选择"""

    # 设置日志
    setup_logging(cfg)
    log = logging.getLogger(__name__)

    # 记录开始时间
    start_time = datetime.now()

    log.info("=== 开始随机数据选择 ===")
    log.info(f"选择比例: {cfg.selection_percentage}")
    log.info(f"随机种子: {cfg.seed}")
    log.info(f"数据源: {cfg.dataset.dataset_from}")

    try:
        # 1. 加载数据集
        log.info("加载数据集...")
        all_data = load_datasets(cfg)

        # 2. 随机选择
        log.info("开始随机选择...")
        selected_data = streaming_random_selection(all_data, cfg.selection_percentage, cfg.seed)

        # 3. 保存选择结果
        log.info("保存选择结果...")
        output_path = os.path.join(os.getcwd(), cfg.output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in selected_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 记录结束时间
        end_time = datetime.now()

        # 4. 保存选择配置
        save_selection_config(output_path, cfg, start_time, end_time)

        log.info(f"随机选择的数据已保存到: {output_path}")
        log.info("=== 随机数据选择完成 ===")

    except Exception as e:
        log.error(f"随机选择过程中发生错误: {e}")
        raise e


if __name__ == "__main__":
    main()
