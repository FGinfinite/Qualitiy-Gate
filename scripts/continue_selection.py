#!/usr/bin/env python3
"""
独立的聚类选择脚本

使用聚类-轮选策略从保存的router_data进行数据选择。
支持K-Means和HDBSCAN两种聚类算法。
现在使用Hydra进行配置管理。
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List

import hydra
import torch
from omegaconf import DictConfig

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.dataset_loader import load_local_datasets
from src.stages.selection import cluster_based_selection, load_router_data
from src.utils.hydra_resolvers import register_custom_resolvers

# Register custom Hydra resolvers before @hydra.main
register_custom_resolvers()


def setup_logging(cfg: DictConfig):
    """设置日志配置"""
    verbose = cfg.logging.get("verbose", False)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def load_all_router_data(router_data_dir: str) -> Dict[str, Dict[str, Any]]:
    """加载所有数据集的router_data文件"""
    log = logging.getLogger(__name__)
    all_router_data = {}

    if not os.path.exists(router_data_dir):
        raise FileNotFoundError(f"Router data目录不存在: {router_data_dir}")

    # 查找所有的router_data文件
    for filename in os.listdir(router_data_dir):
        if filename.endswith("_router_data.pt"):
            dataset_name = filename.replace("_router_data.pt", "")
            file_path = os.path.join(router_data_dir, filename)

            log.info(f"加载数据集 '{dataset_name}' 的router data: {file_path}")
            router_data = load_router_data(file_path)
            all_router_data[dataset_name] = router_data

    if not all_router_data:
        raise ValueError(f"在目录 {router_data_dir} 中未找到任何router_data文件")

    log.info(f"成功加载 {len(all_router_data)} 个数据集的router数据")
    return all_router_data


def load_original_dataset_mapping(router_data_dir: str, data_dir: str = None) -> Dict[str, Dict[str, Any]]:
    """加载原始数据集的消息映射"""
    log = logging.getLogger(__name__)

    # 推断数据集目录
    if data_dir is None:
        current_dir = router_data_dir
        project_root = None
        for _ in range(10):  # 最多向上10层
            current_dir = os.path.dirname(current_dir)
            if os.path.exists(os.path.join(current_dir, "dataset", "train", "processed")):
                project_root = current_dir
                break

        if project_root is None:
            project_root = os.getcwd()

        data_dir = os.path.join(project_root, "dataset", "train", "processed")

    if not os.path.exists(data_dir):
        log.warning(f"推断的数据集目录不存在: {data_dir}")
        return {}

    log.info(f"使用标准数据加载器从目录加载数据: {data_dir}")

    # 获取router_data中的数据集名称
    dataset_names = []
    for filename in os.listdir(router_data_dir):
        if filename.endswith("_router_data.pt"):
            dataset_name = filename.replace("_router_data.pt", "")
            dataset_names.append(dataset_name)

    if not dataset_names:
        log.warning("未找到任何router_data文件")
        return {}

    log.info(f"需要加载的数据集: {dataset_names}")

    try:
        # 使用标准数据加载器加载所有数据集
        combined_dataset = load_local_datasets(
            data_dir=data_dir,
            dataset_names=dataset_names,
            sample_percentage=1.0,  # 加载全部数据
            seed=0,
        )

        log.info(f"标准加载器成功加载 {len(combined_dataset)} 个样本")

        # 将数据按ID组织为映射字典
        dataset_mapping = {name: {} for name in dataset_names}

        for item in combined_dataset:
            dataset_name = item.get("dataset")
            item_id = item.get("id")
            messages = item.get("messages", [])

            if dataset_name and item_id and dataset_name in dataset_mapping:
                dataset_mapping[dataset_name][item_id] = messages

        # 输出加载统计
        for dataset_name in dataset_names:
            count = len(dataset_mapping[dataset_name])
            log.info(f"数据集 '{dataset_name}': 映射了 {count} 个样本")

        return dataset_mapping

    except Exception as e:
        log.error(f"使用标准数据加载器加载失败: {e}")
        return {}


def rebuild_scored_data_with_messages(all_router_data: Dict[str, Dict[str, Any]], dataset_mapping: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从router_data重建scored_data列表，包含完整的messages"""
    log = logging.getLogger(__name__)
    scored_data = []

    for dataset_name, router_data in all_router_data.items():
        quality_score = router_data["quality_score"]  # [N, L, 1]
        sample_ids = router_data["sample_ids"]
        num_samples = router_data["num_samples"]

        log.info(f"处理数据集 '{dataset_name}': {num_samples} 个样本")

        # 获取该数据集的消息映射
        messages_mapping = dataset_mapping.get(dataset_name, {})

        for i in range(num_samples):
            # 从质量门分数计算质量分数
            sample_quality_score = quality_score[i]  # [L, 1]

            # 计算质量分数：对所有层的质量分数求平均
            # 质量分数是sigmoid之后的好概率，直接取平均
            quality_scores = sample_quality_score.squeeze(-1)  # [L]
            final_score = quality_scores.mean().item()

            # 获取原始messages
            sample_id = sample_ids[i]
            messages = messages_mapping.get(sample_id, [])

            if not messages:
                log.warning(f"样本 {sample_id} 未找到对应的messages")

            # 构建scored_data项
            scored_item = {"dataset": dataset_name, "id": sample_id, "scores": final_score, "messages": messages}
            scored_data.append(scored_item)

    log.info(f"成功重建 {len(scored_data)} 个样本的scored_data（包含messages）")
    return scored_data


def rebuild_logits_data(all_router_data: Dict[str, Dict[str, Any]]) -> Dict[str, List[torch.Tensor]]:
    """从router_data重建logits数据用于聚类选择"""
    log = logging.getLogger(__name__)
    all_logits_by_dataset = {}

    for dataset_name, router_data in all_router_data.items():
        moe_logits = router_data["moe_logits"]  # [N, L, E]
        num_samples = router_data["num_samples"]

        # 将张量拆分为列表，每个元素是 [L, E]
        logits_list = [moe_logits[i] for i in range(num_samples)]
        all_logits_by_dataset[dataset_name] = logits_list

        log.info(f"数据集 '{dataset_name}': {len(logits_list)} 个logits张量，形状 {logits_list[0].shape} [L, E]")

    return all_logits_by_dataset


def parse_clustering_params(cfg: DictConfig) -> Dict:
    """解析聚类参数"""
    params = {}
    clustering_params = cfg.clustering_params

    if cfg.clustering_method == "kmeans":
        params.update(
            {
                "auto_k": clustering_params.get("auto_k", True),
                "k": clustering_params.get("k", None),
                "k_range": clustering_params.get("k_range", [10, 100]),
                "max_iters": clustering_params.get("max_iters", 300),
                # 并行计算参数
                "enable_parallel_kmeans": clustering_params.get("enable_parallel_kmeans", False),
                "parallel_processes": clustering_params.get("parallel_processes", 4),
                "gpu_allocation_strategy": clustering_params.get("gpu_allocation_strategy", "round_robin"),
            }
        )
    elif cfg.clustering_method == "hdbscan":
        params.update(
            {
                "min_cluster_size": clustering_params.get("min_cluster_size", None),
                "min_samples": clustering_params.get("min_samples", None),
                "metric": clustering_params.get("metric", "cosine"),
                "use_gpu": clustering_params.get("use_gpu", True),
                "auto_tune": clustering_params.get("auto_tune", False),
            }
        )

    return params


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


@hydra.main(config_path="../configs", config_name="continue_selection", version_base=None)
def main(cfg: DictConfig) -> None:
    """主函数：执行聚类-轮选数据选择"""

    # 验证必需参数
    if cfg.router_data_dir is None:
        raise ValueError("router_data_dir参数是必需的")

    # 设置日志
    setup_logging(cfg)
    log = logging.getLogger(__name__)

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

        log.info(f"聚类选择的数据已保存到: {output_path}")
        log.info("=== 聚类-轮选数据选择完成 ===")

    except Exception as e:
        log.error(f"聚类选择过程中发生错误: {e}")
        raise e


if __name__ == "__main__":
    main()
