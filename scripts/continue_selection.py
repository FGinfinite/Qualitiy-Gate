#!/usr/bin/env python3
"""
独立的聚类选择脚本

使用聚类-轮选策略从保存的router_data进行数据选择。
现在仅支持K-Means聚类算法。
使用Hydra进行配置管理。

注意：从router_data加载的quality_score是原始的质量门分数，
需要应用sigmoid转换为[0,1]范围内的概率值后再计算最终质量分数。
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict

import hydra
import torch
import yaml
from omegaconf import DictConfig

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.selection.data_selection import (
    cluster_based_selection,
    load_all_router_data,
    load_original_dataset_mapping,
    parse_clustering_params,
    rebuild_logits_data,
    rebuild_scored_data_with_messages,
)
from src.utils.hydra_resolvers import register_custom_resolvers

# Register custom Hydra resolvers before @hydra.main
register_custom_resolvers()


def setup_logging(cfg: DictConfig):
    """设置日志配置"""
    verbose = cfg.logging.get("verbose", False)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")












def generate_output_path(router_data_dir: str) -> str:
    """
    根据router_data_dir自动生成output_path
    将输出文件放在与router_data_dir同一父目录下
    """
    # 获取父目录
    parent_dir = os.path.dirname(router_data_dir)
    # 生成输出文件路径
    output_path = os.path.join(parent_dir, "selected_data.jsonl")
    return output_path


def save_selection_config(output_path: str, cfg: DictConfig, clustering_params: Dict, start_time: datetime, end_time: datetime, device: torch.device):
    """保存数据选择配置到selection_config.yaml"""
    log = logging.getLogger(__name__)

    # 构建配置字典
    config_dict = {
        "selection_metadata": {
            "script_name": "continue_selection.py",
            "execution_time": {"start": start_time.isoformat(), "end": end_time.isoformat(), "duration_seconds": (end_time - start_time).total_seconds()},
            "device_used": str(device),
        },
        "selection_parameters": {
            "selection_percentage": float(cfg.selection_percentage),
            "clustering_method": cfg.clustering_method,
            "debug_print": cfg.debug_print,
        },
        "clustering_parameters": clustering_params,
        "paths": {"router_data_dir": cfg.router_data_dir, "selected_data_path": output_path, "data_dir": cfg.get("data_dir", "auto-inferred")},
        "device_configuration": {
            "device_type": device.type,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
    }

    # 保存配置文件
    config_path = os.path.join(os.path.dirname(output_path), "selection_config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    log.info(f"选择配置已保存到: {config_path}")


@hydra.main(config_path="../configs", config_name="continue_selection", version_base=None)
def main(cfg: DictConfig) -> None:
    """主函数：执行聚类-轮选数据选择"""

    # 验证必需参数
    if cfg.router_data_dir is None:
        raise ValueError("router_data_dir参数是必需的")

    # 设置日志
    setup_logging(cfg)
    log = logging.getLogger(__name__)

    # 记录开始时间
    start_time = datetime.now()

    log.info("=== 开始聚类-轮选数据选择 ===")
    log.info(f"聚类方法: {cfg.clustering_method}")
    log.info(f"选择比例: {cfg.selection_percentage}")

    # 确定设备
    if cfg.device == "auto":
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                device = torch.device("cuda")
                log.info(f"使用设备: {device} (实际GPU: {torch.cuda.current_device()}, 共{num_gpus}个可用设备)")
            else:
                device = torch.device("cpu")
                log.info("没有可用的GPU设备，使用CPU")
        else:
            device = torch.device("cpu")
            log.info("CUDA不可用，使用CPU")
    else:
        device = torch.device(cfg.device)
        log.info(f"使用指定设备: {device}")

    # 如果使用GPU，清理缓存
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 自动生成输出路径
    output_path = generate_output_path(cfg.router_data_dir)
    log.info(f"输出路径: {output_path}")

    try:
        # 1. 加载所有router_data
        log.info("加载router_data文件...")
        all_router_data = load_all_router_data(cfg.router_data_dir)

        # 2. 加载原始数据集映射
        log.info("加载原始数据集映射...")
        dataset_mapping = load_original_dataset_mapping(cfg.router_data_dir, cfg.data_dir)

        # 3. 重建scored_data（包含messages）
        log.info("重建scored_data（包含完整messages）...")
        scored_data = rebuild_scored_data_with_messages(all_router_data, dataset_mapping)

        # 4. 重建logits数据
        log.info("重建logits数据...")
        all_logits_by_dataset = rebuild_logits_data(all_router_data)

        # 5. 解析聚类参数
        clustering_params = parse_clustering_params(cfg)
        log.info(f"聚类参数: {clustering_params}")

        # 6. 执行聚类-轮选选择
        log.info("开始聚类-轮选数据选择...")
        selected_data = cluster_based_selection(
            scored_data=scored_data,
            all_logits_by_dataset=all_logits_by_dataset,
            selection_percentage=cfg.selection_percentage,
            clustering_method=cfg.clustering_method,
            clustering_params=clustering_params,
            device=device,
            debug_print=cfg.debug_print,
            output_dir=os.path.dirname(output_path),
        )

        log.info(f"聚类-轮选完成，选择了 {len(selected_data)} 个样本 ({cfg.selection_percentage * 100:.2f}%)")

        if len(selected_data) > 3:
            log.info(f"前3个分数: {[d['scores'] for d in selected_data[:3]]}")
            log.info(f"后3个分数: {[d['scores'] for d in selected_data[-3:]]}")

        # 7. 保存选择结果
        log.info("保存选择结果...")
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in selected_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 记录结束时间
        end_time = datetime.now()

        # 8. 保存选择配置
        save_selection_config(output_path, cfg, clustering_params, start_time, end_time, device)

        log.info(f"聚类选择的数据已保存到: {output_path}")
        log.info("=== 聚类-轮选数据选择完成 ===")

    except Exception as e:
        log.error(f"聚类选择过程中发生错误: {e}")
        raise e


if __name__ == "__main__":
    main()
