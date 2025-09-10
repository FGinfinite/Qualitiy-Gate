# src/selection/data_selection.py

import logging
import os
from typing import Any, Dict, List

import torch
from omegaconf import DictConfig

from src.clustering import ClusterBasedSelection
from src.data.dataset_loader import load_hf_datasets, load_local_datasets


def cluster_based_selection(
    scored_data: List[dict],
    all_logits_by_dataset: dict,
    selection_percentage: float,
    clustering_method: str = "kmeans",
    clustering_params: dict = None,
    device: torch.device = None,
    debug_print: bool = False,
    output_dir: str = None,
) -> List[dict]:
    """
    基于聚类的数据选择策略

    Args:
        scored_data: 评分后的数据列表
        all_logits_by_dataset: 按数据集分组的logits张量
        selection_percentage: 选择比例
        clustering_method: 聚类方法 ('kmeans' 或 'hdbscan')
        clustering_params: 聚类参数
        device: GPU设备
        debug_print: 是否启用调试输出
        output_dir: 输出目录，用于创建子进程日志文件

    Returns:
        选择后的数据列表
    """
    total_samples = len(scored_data)
    target_count = int(total_samples * selection_percentage)

    log = logging.getLogger(__name__)
    log.info(f"开始聚类选择: 从 {total_samples} 个样本中选择 {target_count} 个")
    log.info(f"使用聚类方法: {clustering_method}")

    if debug_print:
        log.info("启用调试模式")

    # 初始化聚类选择器
    cluster_selector = ClusterBasedSelection(device=device, debug_print=debug_print, output_dir=output_dir)

    # 执行聚类-轮选选择
    selected_data = cluster_selector.select_data_by_clustering(
        scored_data=scored_data,
        all_logits_by_dataset=all_logits_by_dataset,
        target_count=target_count,
        clustering_method=clustering_method,
        clustering_params=clustering_params or {},
    )

    return selected_data


def load_all_router_data(router_data_dir: str) -> Dict[str, Dict[str, Any]]:
    """加载所有数据集的router_data文件"""
    from src.stages.selection import load_router_data

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


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
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


def load_original_dataset_mapping(router_data_dir: str, data_dir: str = None) -> Dict[str, Dict[str, Any]]:
    """
    使用工厂函数加载原始数据集的消息映射

    Args:
        router_data_dir: router_data文件目录
        data_dir: 数据目录路径（如果指定，会覆盖默认推断）

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

    log.info(f"需要加载的数据集: {dataset_names}")

    # 根据数据集名称确定加载策略
    dataset_mapping = {}

    for dataset_name in dataset_names:
        log.info(f"处理数据集: {dataset_name}")

        # 获取数据集配置
        dataset_config = get_dataset_config(dataset_name)

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

                log.info(f"从本地加载数据集 '{dataset_name}': {effective_data_dir}")
                combined_dataset = load_local_datasets(data_dir=effective_data_dir, dataset_names=config["dataset_names"], sample_percentage=1.0, seed=0)

            elif source_type == "hf":
                # 加载HuggingFace数据集
                log.info(f"从HuggingFace加载数据集 '{dataset_name}'")
                combined_dataset = load_hf_datasets(hf_config={"datasets": config["datasets"]}, sample_percentage=1.0, seed=0)
            else:
                log.error(f"不支持的数据源类型: {source_type}")
                dataset_mapping[dataset_name] = {}
                continue

            # 组织数据映射
            single_dataset_mapping = _organize_dataset_mapping(combined_dataset, [dataset_name], log)
            dataset_mapping.update(single_dataset_mapping)

        except Exception as e:
            log.error(f"加载数据集 '{dataset_name}' 失败: {e}")
            dataset_mapping[dataset_name] = {}

    return dataset_mapping


def _organize_dataset_mapping(combined_dataset, dataset_names: List[str], log) -> Dict[str, Dict[str, Any]]:
    """将数据按ID组织为映射字典"""
    dataset_mapping = {name: {} for name in dataset_names}

    for item in combined_dataset:
        dataset_name = item.get("dataset")
        item_id = item.get("id")
        messages = item.get("messages", [])

        if dataset_name and item_id and dataset_name in dataset_mapping:
            dataset_mapping[dataset_name][item_id] = messages

    # 输出加载统计
    for dataset_name, mapping in dataset_mapping.items():
        log.info(f"数据集 '{dataset_name}' 包含 {len(mapping)} 个样本映射")

    return dataset_mapping


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

            # 计算质量分数：先对原始分数应用sigmoid，再求平均
            # 注意：从router_data加载的quality_score是原始分数，需要sigmoid转换为概率
            quality_scores_raw = sample_quality_score.squeeze(-1)  # [L]
            quality_scores_sigmoid = torch.sigmoid(quality_scores_raw)  # [L] sigmoid转换为[0,1]概率
            final_score = quality_scores_sigmoid.mean().item()

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
    else:
        # 为未来扩展保留接口，目前仅支持kmeans
        supported_methods = ["kmeans"]
        raise ValueError(f"不支持的聚类方法: {cfg.clustering_method}。支持的方法: {supported_methods}")

    return params
