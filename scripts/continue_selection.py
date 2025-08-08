#!/usr/bin/env python3
"""
独立的数据选择继续脚本

用于在模型推理完成并保存router_data后，从保存的张量继续执行数据选择过程。
主要用于解决GPU内存不足导致距离计算失败的问题。
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

import torch

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.stages.selection import diversity_based_selection, load_router_data
from src.data.dataset_loader import load_local_datasets


def setup_logging(verbose: bool = False):
    """设置日志配置"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_all_router_data(router_data_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    加载所有数据集的router_data文件

    Args:
        router_data_dir: router_data目录路径

    Returns:
        包含所有数据集router_data的字典
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

            log.info(f"加载数据集 '{dataset_name}' 的router data: {file_path}")
            router_data = load_router_data(file_path)
            all_router_data[dataset_name] = router_data

    if not all_router_data:
        raise ValueError(f"在目录 {router_data_dir} 中未找到任何router_data文件")

    log.info(f"成功加载 {len(all_router_data)} 个数据集的router数据")
    return all_router_data


def load_original_dataset_mapping(router_data_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    从router_data目录推断原始数据集路径并使用标准加载器加载数据映射

    Args:
        router_data_dir: router_data目录路径

    Returns:
        原始数据集的ID到消息映射字典
    """
    log = logging.getLogger(__name__)

    # 从router_data_dir推断项目根目录和数据集目录
    current_dir = router_data_dir
    project_root = None
    for _ in range(10):  # 最多向上10层
        current_dir = os.path.dirname(current_dir)
        if os.path.exists(os.path.join(current_dir, "dataset", "train", "processed")):
            project_root = current_dir
            break

    if project_root is None:
        # 如果没找到，使用当前工作目录
        project_root = os.getcwd()

    dataset_dir = os.path.join(project_root, "dataset", "train", "processed")

    if not os.path.exists(dataset_dir):
        log.warning(f"推断的数据集目录不存在: {dataset_dir}")
        return {}

    log.info(f"使用标准数据加载器从目录加载数据: {dataset_dir}")

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
            data_dir=dataset_dir,
            dataset_names=dataset_names,
            sample_percentage=1.0,  # 加载全部数据
            seed=0
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
        # 回退到原来的手动加载方式
        log.info("回退到手动文件加载方式...")

        dataset_mapping = {}
        for dataset_name in dataset_names:
            dataset_subdir = os.path.join(dataset_dir, dataset_name)
            if os.path.exists(dataset_subdir):
                data_files = []
                for file in os.listdir(dataset_subdir):
                    if file.endswith(".jsonl"):
                        data_files.append(os.path.join(dataset_subdir, file))

                if data_files:
                    dataset_mapping[dataset_name] = {}
                    for data_file in data_files:
                        try:
                            log.info(f"手动加载数据文件: {data_file}")
                            with open(data_file, "r", encoding="utf-8") as f:
                                for line in f:
                                    if line.strip():
                                        item = json.loads(line)
                                        item_id = item.get("id")
                                        if item_id:
                                            dataset_mapping[dataset_name][item_id] = item.get("messages", [])
                            log.info(f"从 {data_file} 加载了 {len(dataset_mapping[dataset_name])} 个样本")
                        except Exception as file_e:
                            log.error(f"手动加载文件 {data_file} 失败: {file_e}")
                else:
                    log.warning(f"在 {dataset_subdir} 中未找到任何JSONL文件")
                    dataset_mapping[dataset_name] = {}
            else:
                log.warning(f"数据集目录不存在: {dataset_subdir}")
                dataset_mapping[dataset_name] = {}

        return dataset_mapping


def rebuild_scored_data_with_messages(
    all_router_data: Dict[str, Dict[str, Any]], dataset_mapping: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    从router_data重建scored_data列表，包含完整的messages

    Args:
        all_router_data: 所有数据集的router数据
        dataset_mapping: 原始数据集的消息映射

    Returns:
        重建的scored_data列表（包含messages）
    """
    log = logging.getLogger(__name__)
    scored_data = []

    for dataset_name, router_data in all_router_data.items():
        quality_logits = router_data["quality_logits"]  # [N, L, 2]
        sample_ids = router_data["sample_ids"]
        num_samples = router_data["num_samples"]

        log.info(f"处理数据集 '{dataset_name}': {num_samples} 个样本")

        # 获取该数据集的消息映射
        messages_mapping = dataset_mapping.get(dataset_name, {})

        for i in range(num_samples):
            # 从质量门logits计算质量分数
            sample_quality_logits = quality_logits[i]  # [L, 2]

            # 计算质量分数：对所有层的good概率求平均
            quality_probs = torch.softmax(sample_quality_logits, dim=-1)  # [L, 2]
            good_probs = quality_probs[:, 0]  # [L]
            quality_score = good_probs.mean().item()

            # 获取原始messages
            sample_id = sample_ids[i]
            messages = messages_mapping.get(sample_id, [])

            if not messages:
                log.warning(f"样本 {sample_id} 未找到对应的messages")

            # 构建scored_data项
            scored_item = {"dataset": dataset_name, "id": sample_id, "scores": quality_score, "messages": messages}
            scored_data.append(scored_item)

    log.info(f"成功重建 {len(scored_data)} 个样本的scored_data（包含messages）")
    return scored_data


def rebuild_scored_data(all_router_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    从router_data重建scored_data列表

    Args:
        all_router_data: 所有数据集的router数据

    Returns:
        重建的scored_data列表
    """
    log = logging.getLogger(__name__)
    scored_data = []

    for dataset_name, router_data in all_router_data.items():
        quality_logits = router_data["quality_logits"]  # [N, L, 2]
        sample_ids = router_data["sample_ids"]
        num_samples = router_data["num_samples"]

        log.info(f"处理数据集 '{dataset_name}': {num_samples} 个样本")

        for i in range(num_samples):
            # 从质量门logits计算质量分数
            # quality_logits[i]: [L, 2] - 已经是平均过的概率
            sample_quality_logits = quality_logits[i]  # [L, 2]

            # 计算质量分数：对所有层的good概率求平均
            # 已经是平均过的概率，直接使用softmax
            quality_probs = torch.softmax(sample_quality_logits, dim=-1)  # [L, 2]
            good_probs = quality_probs[:, 0]  # [L]
            quality_score = good_probs.mean().item()

            # 构建scored_data项
            scored_item = {
                "dataset": dataset_name,
                "id": sample_ids[i],
                "scores": quality_score,
                "messages": [],  # 空的messages，因为我们只需要进行多样性选择
            }
            scored_data.append(scored_item)

    log.info(f"成功重建 {len(scored_data)} 个样本的scored_data")
    return scored_data


def rebuild_logits_data(all_router_data: Dict[str, Dict[str, Any]]) -> Dict[str, List[torch.Tensor]]:
    """
    从router_data重建logits数据用于多样性选择

    Args:
        all_router_data: 所有数据集的router数据

    Returns:
        按数据集分组的moe_logits列表
    """
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


def main():
    parser = argparse.ArgumentParser(description="从保存的router_data继续数据选择过程")
    parser.add_argument("--router_data_dir", required=True, help="包含router_data文件的目录路径")
    parser.add_argument("--output_path", required=True, help="选择结果的输出文件路径")
    parser.add_argument("--selection_percentage", type=float, required=True, help="数据选择比例")
    parser.add_argument("--importance_selection_percentage", type=float, help="两阶段选择的第一阶段比例（可选）")
    parser.add_argument("--disable_diversity", action="store_true", help="禁用多样性选择，使用质量分数选择")
    parser.add_argument("--distance_batch_size", type=int, default=500, help="距离计算的批处理大小（默认: 500）")
    parser.add_argument("--fps_log_interval", type=int, default=100, help="FPS日志输出间隔（默认: 100）")
    parser.add_argument("--device", default="auto", help="计算设备 (cuda/cpu/auto，默认: auto)")
    parser.add_argument("--verbose", "-v", action="store_true", help="启用详细日志输出")

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.verbose)
    log = logging.getLogger(__name__)

    log.info("=== 开始从router_data继续数据选择 ===")

    # 确定设备
    if args.device == "auto":
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
        device = torch.device(args.device)
        log.info(f"使用指定设备: {device}")

    # 如果使用GPU，清理缓存
    if device.type == "cuda":
        torch.cuda.empty_cache()

    try:
        # 1. 加载所有router_data
        log.info("加载router_data文件...")
        all_router_data = load_all_router_data(args.router_data_dir)

        # 1.5. 加载原始数据集映射
        log.info("加载原始数据集映射...")
        dataset_mapping = load_original_dataset_mapping(args.router_data_dir)

        # 2. 重建scored_data（包含messages）
        log.info("重建scored_data（包含完整messages）...")
        scored_data = rebuild_scored_data_with_messages(all_router_data, dataset_mapping)

        # 3. 重建logits数据
        log.info("重建logits数据...")
        all_logits_by_dataset = rebuild_logits_data(all_router_data)

        # 4. 执行数据选择
        log.info("开始数据选择...")
        enable_diversity = not args.disable_diversity
        log.info(f"多样性选择模式: {'启用' if enable_diversity else '禁用(使用质量分数)'}")

        selected_data = diversity_based_selection(
            scored_data=scored_data,
            all_logits_by_dataset=all_logits_by_dataset,
            selection_percentage=args.selection_percentage,
            importance_selection_percentage=args.importance_selection_percentage,
            enable_diversity=enable_diversity,
            device=device,
            distance_batch_size=args.distance_batch_size,
            fps_log_interval=args.fps_log_interval,
        )

        log.info(f"选择了前 {len(selected_data)} 个样本 ({args.selection_percentage * 100:.2f}%)")

        if len(selected_data) > 3:
            log.info(f"前3个分数: {[d['scores'] for d in selected_data[:3]]}")
            log.info(f"后3个分数: {[d['scores'] for d in selected_data[-3:]]}")

        # 5. 保存选择结果
        log.info("保存选择结果...")
        output_dir = os.path.dirname(args.output_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(args.output_path, "w", encoding="utf-8") as f:
            for item in selected_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        log.info(f"筛选后的数据已保存到: {args.output_path}")
        log.info("=== 数据选择完成 ===")

    except Exception as e:
        log.error(f"数据选择过程中发生错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
