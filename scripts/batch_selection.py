#!/usr/bin/env python3
"""
自动化批量数据选择脚本

递归扫描指定目录下的所有实验输出，对每个包含router_data的目录执行聚类选择。
支持跳过已经有选好数据的目录，并在完成后保存选择配置。
"""

import json
import logging
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import yaml
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


def find_experiment_directories(root_dir: str) -> List[Tuple[str, str]]:
    """
    递归查找所有包含router_data的实验目录

    返回: [(experiment_dir, router_data_dir), ...]
    """
    log = logging.getLogger(__name__)
    experiment_dirs = []

    if not os.path.exists(root_dir):
        log.error(f"根目录不存在: {root_dir}")
        return experiment_dirs

    log.info(f"扫描实验目录: {root_dir}")

    # 递归查找所有router_data目录
    for root, dirs, _files in os.walk(root_dir):
        if "router_data" in dirs:
            router_data_dir = os.path.join(root, "router_data")

            # 检查router_data目录是否包含有效的数据文件
            if is_valid_router_data_dir(router_data_dir):
                experiment_dirs.append((root, router_data_dir))
                log.debug(f"发现有效实验目录: {root}")

    log.info(f"找到 {len(experiment_dirs)} 个有效的实验目录")
    return experiment_dirs


def is_valid_router_data_dir(router_data_dir: str) -> bool:
    """检查router_data目录是否包含有效的数据文件"""
    if not os.path.exists(router_data_dir):
        return False

    # 查找router_data文件
    router_data_files = [f for f in os.listdir(router_data_dir) if f.endswith("_router_data.pt")]
    return len(router_data_files) > 0


def should_skip_experiment(experiment_dir: str, skip_existing: bool) -> Tuple[bool, str]:
    """
    判断是否应该跳过某个实验目录

    返回: (should_skip, reason)
    """
    selected_data_path = os.path.join(experiment_dir, "selected_data.jsonl")
    selection_config_path = os.path.join(experiment_dir, "selection_config.yaml")

    if skip_existing:
        if os.path.exists(selected_data_path):
            if os.path.exists(selection_config_path):
                return True, "已存在 selected_data.jsonl 和 selection_config.yaml"
            else:
                return True, "已存在 selected_data.jsonl（但缺少 selection_config.yaml）"

    return False, ""


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

            log.debug(f"加载数据集 '{dataset_name}' 的router data: {file_path}")
            router_data = load_router_data(file_path)
            all_router_data[dataset_name] = router_data

    if not all_router_data:
        raise ValueError(f"在目录 {router_data_dir} 中未找到任何router_data文件")

    log.debug(f"成功加载 {len(all_router_data)} 个数据集的router数据")
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

    log.debug(f"使用标准数据加载器从目录加载数据: {data_dir}")

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

    try:
        # 使用标准数据加载器加载所有数据集
        combined_dataset = load_local_datasets(
            data_dir=data_dir,
            dataset_names=dataset_names,
            sample_percentage=1.0,  # 加载全部数据
            seed=0,
        )

        log.debug(f"标准加载器成功加载 {len(combined_dataset)} 个样本")

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
            log.debug(f"数据集 '{dataset_name}': 映射了 {count} 个样本")

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

        log.debug(f"处理数据集 '{dataset_name}': {num_samples} 个样本")

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

    log.debug(f"成功重建 {len(scored_data)} 个样本的scored_data（包含messages）")
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

        log.debug(f"数据集 '{dataset_name}': {len(logits_list)} 个logits张量，形状 {logits_list[0].shape} [L, E]")

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


def save_selection_config(experiment_dir: str, cfg: DictConfig, clustering_params: Dict, start_time: datetime, end_time: datetime, device: torch.device):
    """保存数据选择配置到selection_config.yaml"""
    log = logging.getLogger(__name__)

    # 构建配置字典
    config_dict = {
        "selection_metadata": {
            "script_name": "batch_selection.py",
            "execution_time": {"start": start_time.isoformat(), "end": end_time.isoformat(), "duration_seconds": (end_time - start_time).total_seconds()},
            "device_used": str(device),
        },
        "selection_parameters": {
            "selection_percentage": float(cfg.selection_percentage),
            "clustering_method": cfg.clustering_method,
            "debug_print": cfg.debug_print,
        },
        "clustering_parameters": clustering_params,
        "paths": {
            "experiment_dir": experiment_dir,
            "router_data_dir": os.path.join(experiment_dir, "router_data"),
            "selected_data_path": os.path.join(experiment_dir, "selected_data.jsonl"),
            "data_dir": cfg.get("data_dir", "auto-inferred"),
        },
        "device_configuration": {
            "device_type": device.type,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
    }

    # 保存配置文件
    config_path = os.path.join(experiment_dir, "selection_config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    log.info(f"选择配置已保存到: {config_path}")


def backup_existing_files(experiment_dir: str) -> List[str]:
    """备份现有的输出文件"""
    log = logging.getLogger(__name__)
    backup_files = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    files_to_backup = ["selected_data.jsonl", "selection_config.yaml"]

    for filename in files_to_backup:
        file_path = os.path.join(experiment_dir, filename)
        if os.path.exists(file_path):
            backup_path = os.path.join(experiment_dir, f"{filename}.backup_{timestamp}")
            shutil.copy2(file_path, backup_path)
            backup_files.append(backup_path)
            log.info(f"已备份 {filename} 到 {backup_path}")

    return backup_files


def process_single_experiment(experiment_dir: str, router_data_dir: str, cfg: DictConfig, device: torch.device) -> Tuple[bool, str, Optional[Dict]]:
    """
    处理单个实验目录

    返回: (success, message, stats)
    """
    log = logging.getLogger(__name__)
    start_time = datetime.now()

    try:
        log.info(f"开始处理实验: {experiment_dir}")

        # 如果启用强制处理，先备份现有文件
        if cfg.force_processing.get("enabled", False) and cfg.force_processing.get("backup_existing", True):
            backup_files = backup_existing_files(experiment_dir)
            log.info(f"备份了 {len(backup_files)} 个文件")

        # 清理GPU缓存
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # 1. 加载所有router_data
        log.debug("加载router_data文件...")
        all_router_data = load_all_router_data(router_data_dir)

        # 2. 加载原始数据集映射
        log.debug("加载原始数据集映射...")
        dataset_mapping = load_original_dataset_mapping(router_data_dir, cfg.data_dir)

        # 3. 重建scored_data（包含messages）
        log.debug("重建scored_data（包含完整messages）...")
        scored_data = rebuild_scored_data_with_messages(all_router_data, dataset_mapping)

        # 4. 重建logits数据
        log.debug("重建logits数据...")
        all_logits_by_dataset = rebuild_logits_data(all_router_data)

        # 5. 解析聚类参数
        clustering_params = parse_clustering_params(cfg)
        log.debug(f"聚类参数: {clustering_params}")

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
            output_dir=experiment_dir,
        )

        # 7. 保存选择结果
        output_path = os.path.join(experiment_dir, "selected_data.jsonl")
        os.makedirs(experiment_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in selected_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        end_time = datetime.now()

        # 8. 保存选择配置
        save_selection_config(experiment_dir, cfg, clustering_params, start_time, end_time, device)

        # 统计信息
        stats = {
            "selected_count": len(selected_data),
            "total_samples": len(scored_data),
            "selection_percentage": cfg.selection_percentage,
            "duration_seconds": (end_time - start_time).total_seconds(),
            "datasets_processed": len(all_router_data),
        }

        log.info(f"实验处理完成: {experiment_dir}")
        log.info(f"选择了 {len(selected_data)} / {len(scored_data)} 个样本 ({cfg.selection_percentage * 100:.2f}%)")

        return True, f"成功处理 {len(selected_data)} / {len(scored_data)} 个样本", stats

    except Exception as e:
        end_time = datetime.now()
        error_msg = f"处理失败: {str(e)}"
        log.error(f"实验处理失败 {experiment_dir}: {e}", exc_info=True)
        return False, error_msg, None


def generate_processing_report(processed_experiments: List[Dict], skipped_experiments: List[Dict], failed_experiments: List[Dict], cfg: DictConfig) -> str:
    """生成处理报告"""
    report = {
        "summary": {
            "total_experiments_found": len(processed_experiments) + len(skipped_experiments) + len(failed_experiments),
            "processed_successfully": len(processed_experiments),
            "skipped": len(skipped_experiments),
            "failed": len(failed_experiments),
            "timestamp": datetime.now().isoformat(),
        },
        "processed_experiments": processed_experiments,
        "skipped_experiments": skipped_experiments,
        "failed_experiments": failed_experiments,
    }

    # 保存详细报告到当前hydra输出目录
    report_path = os.path.join(cfg.output_dir, "batch_processing_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report_path


@hydra.main(config_path="../configs", config_name="batch_selection", version_base=None)
def main(cfg: DictConfig) -> None:
    """主函数：批量执行聚类-轮选数据选择"""

    # 设置日志
    setup_logging(cfg)
    log = logging.getLogger(__name__)

    log.info("=== 开始批量聚类-轮选数据选择 ===")
    log.info(f"根目录: {cfg.root_dir}")
    log.info(f"聚类方法: {cfg.clustering_method}")
    log.info(f"选择比例: {cfg.selection_percentage}")
    log.info(f"跳过已存在: {cfg.batch_processing.skip_existing}")
    log.info(f"干运行模式: {cfg.batch_processing.dry_run}")

    # 确定设备
    if cfg.device == "auto":
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                device = torch.device("cuda")
                log.info(f"使用设备: {device} (GPU: {torch.cuda.current_device()}, 共{num_gpus}个可用设备)")
            else:
                device = torch.device("cpu")
                log.info("没有可用的GPU设备，使用CPU")
        else:
            device = torch.device("cpu")
            log.info("CUDA不可用，使用CPU")
    else:
        device = torch.device(cfg.device)
        log.info(f"使用指定设备: {device}")

    # 查找所有实验目录
    experiment_dirs = find_experiment_directories(cfg.root_dir)
    if not experiment_dirs:
        log.error("未找到任何有效的实验目录")
        return

    log.info(f"找到 {len(experiment_dirs)} 个实验目录")

    # 处理统计
    processed_experiments = []
    skipped_experiments = []
    failed_experiments = []

    # 逐一处理每个实验
    for i, (experiment_dir, router_data_dir) in enumerate(experiment_dirs, 1):
        log.info(f"\n[{i}/{len(experiment_dirs)}] 检查实验目录: {experiment_dir}")

        # 检查是否应该跳过
        should_skip, skip_reason = should_skip_experiment(experiment_dir, cfg.batch_processing.skip_existing and not cfg.force_processing.get("enabled", False))

        if should_skip:
            log.info(f"跳过实验: {skip_reason}")
            skipped_experiments.append({"experiment_dir": experiment_dir, "reason": skip_reason})
            continue

        # 干运行模式
        if cfg.batch_processing.dry_run:
            log.info(f"[干运行] 将处理实验: {experiment_dir}")
            processed_experiments.append({"experiment_dir": experiment_dir, "status": "dry_run", "message": "干运行模式，未实际处理"})
            continue

        # 实际处理实验
        success, message, stats = process_single_experiment(experiment_dir, router_data_dir, cfg, device)

        if success:
            processed_experiments.append({"experiment_dir": experiment_dir, "status": "success", "message": message, "stats": stats})
        else:
            failed_experiments.append({"experiment_dir": experiment_dir, "status": "failed", "message": message})

            # 如果配置为遇到错误时继续处理
            if not cfg.batch_processing.continue_on_error:
                log.error("遇到错误且未配置继续处理，停止执行")
                break

    # 生成并保存处理报告
    if cfg.batch_processing.save_report:
        report_path = generate_processing_report(processed_experiments, skipped_experiments, failed_experiments, cfg)
        log.info(f"处理报告已保存到: {report_path}")

    # 输出最终统计
    log.info("\n=== 批量处理完成 ===")
    log.info(f"总实验数量: {len(experiment_dirs)}")
    log.info(f"成功处理: {len(processed_experiments)}")
    log.info(f"跳过: {len(skipped_experiments)}")
    log.info(f"失败: {len(failed_experiments)}")

    if processed_experiments:
        total_samples_selected = sum(exp.get("stats", {}).get("selected_count", 0) for exp in processed_experiments if exp.get("stats"))
        total_samples_processed = sum(exp.get("stats", {}).get("total_samples", 0) for exp in processed_experiments if exp.get("stats"))
        log.info(f"总共选择样本数: {total_samples_selected}")
        log.info(f"总共处理样本数: {total_samples_processed}")


if __name__ == "__main__":
    main()
