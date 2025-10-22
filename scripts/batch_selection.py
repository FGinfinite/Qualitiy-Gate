#!/usr/bin/env python3
"""
自动化批量数据选择脚本

递归扫描指定目录下的所有实验输出，对每个包含router_data的目录执行质量选择。
支持跳过已经有选好数据的目录，并在完成后保存选择配置。
"""

import json
import logging
import os
import shutil
import sys
from datetime import datetime
from typing import List, Optional, Tuple

import hydra
from omegaconf import DictConfig

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.selection import (
    load_all_router_data,
    load_original_dataset_mapping,
    prepare_selection_data,
    quality_based_selection,
    save_selection_config,
)
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
            log.debug(f"已备份 {filename} 到 {backup_path}")
    
    return backup_files


def process_single_experiment(experiment_dir: str, router_data_dir: str, cfg: DictConfig) -> Tuple[bool, str, Optional[dict]]:
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
            log.debug(f"备份了 {len(backup_files)} 个文件")
        
        # 1. 加载所有router_data
        log.debug("加载router_data文件...")
        all_router_data = load_all_router_data(router_data_dir)
        
        # 2. 加载原始数据集映射
        log.debug("加载原始数据集映射...")
        dataset_mapping = load_original_dataset_mapping(router_data_dir, cfg.data_dir)
        
        # 3. 准备数据
        log.debug("准备数据...")
        all_quality_gates, all_perplexities, all_sample_ids, all_messages, all_dataset_names = prepare_selection_data(
            all_router_data, dataset_mapping
        )
        
        # 4. 执行质量选择
        log.info("开始基于质量的数据选择...")
        selected_data = quality_based_selection(
            all_quality_gates=all_quality_gates,
            all_perplexities=all_perplexities,
            sample_ids=all_sample_ids,
            messages_list=all_messages,
            dataset_names=all_dataset_names,
            selection_percentage=cfg.selection_percentage,
            alpha=cfg.quality_params.get("alpha", 1.0),
            eps=cfg.quality_params.get("eps", 1e-8),
            tau=cfg.quality_params.get("tau", 0.0),
        )
        
        # 5. 保存选择结果
        output_path = os.path.join(experiment_dir, "selected_data.jsonl")
        os.makedirs(experiment_dir, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for item in selected_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        end_time = datetime.now()
        
        # 6. 保存选择配置
        save_selection_config(
            output_dir=experiment_dir,
            script_name="batch_selection.py",
            selection_percentage=cfg.selection_percentage,
            alpha=cfg.quality_params.get("alpha", 1.0),
            eps=cfg.quality_params.get("eps", 1e-8),
            tau=cfg.quality_params.get("tau", 0.0),
            router_data_dir=router_data_dir,
            selected_data_path=output_path,
            data_dir=cfg.get("data_dir"),
            start_time=start_time,
            end_time=end_time,
        )
        
        # 统计信息
        stats = {
            "selected_count": len(selected_data),
            "total_samples": len(all_sample_ids),
            "selection_percentage": cfg.selection_percentage,
            "duration_seconds": (end_time - start_time).total_seconds(),
            "datasets_processed": len(all_router_data),
        }
        
        log.info(f"实验处理完成: {experiment_dir}")
        log.info(f"选择了 {len(selected_data)} / {len(all_sample_ids)} 个样本 ({cfg.selection_percentage * 100:.2f}%)")
        
        return True, f"成功处理 {len(selected_data)} / {len(all_sample_ids)} 个样本", stats
        
    except Exception as e:
        end_time = datetime.now()
        error_msg = f"处理失败: {str(e)}"
        log.error(f"实验处理失败 {experiment_dir}: {e}", exc_info=True)
        return False, error_msg, None


def generate_processing_report(processed_experiments: List[dict], skipped_experiments: List[dict], failed_experiments: List[dict], cfg: DictConfig) -> str:
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
    """主函数：批量执行基于质量的数据选择"""
    
    # 设置日志
    setup_logging(cfg)
    log = logging.getLogger(__name__)
    
    log.info("=== 开始批量质量数据选择 ===")
    log.info(f"根目录: {cfg.root_dir}")
    log.info(f"选择比例: {cfg.selection_percentage}")
    log.info(f"跳过已存在: {cfg.batch_processing.skip_existing}")
    log.info(f"干运行模式: {cfg.batch_processing.dry_run}")
    
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
        success, message, stats = process_single_experiment(experiment_dir, router_data_dir, cfg)
        
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
