#!/usr/bin/env python3
"""
独立的质量选择脚本

使用质量门控 + 困惑度进行数据选择。
使用Hydra进行配置管理。
"""

import json
import logging
import os
import sys
from datetime import datetime

import hydra
from omegaconf import DictConfig

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.selection import (
    generate_output_path,
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


@hydra.main(config_path="../configs", config_name="continue_selection", version_base=None)
def main(cfg: DictConfig) -> None:
    """主函数：执行基于质量的数据选择"""
    
    # 验证必需参数
    if cfg.router_data_dir is None:
        raise ValueError("router_data_dir参数是必需的")
    
    # 设置日志
    setup_logging(cfg)
    log = logging.getLogger(__name__)
    
    # 记录开始时间
    start_time = datetime.now()
    
    log.info("=== 开始基于质量的数据选择 ===")
    log.info(f"选择比例: {cfg.selection_percentage}")
    
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
        
        # 3. 准备数据
        log.info("准备数据...")
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
        
        log.info(f"质量选择完成，选择了 {len(selected_data)} 个样本 ({cfg.selection_percentage * 100:.2f}%)")
        
        if len(selected_data) > 3:
            log.info(f"前3个分数: {[d['scores'] for d in selected_data[:3]]}")
            log.info(f"后3个分数: {[d['scores'] for d in selected_data[-3:]]}")
        
        # 5. 保存选择结果
        log.info("保存选择结果...")
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for item in selected_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # 记录结束时间
        end_time = datetime.now()
        
        # 6. 保存选择配置
        save_selection_config(
            output_dir=os.path.dirname(output_path),
            script_name="continue_selection.py",
            selection_percentage=cfg.selection_percentage,
            alpha=cfg.quality_params.get("alpha", 1.0),
            eps=cfg.quality_params.get("eps", 1e-8),
            tau=cfg.quality_params.get("tau", 0.0),
            router_data_dir=cfg.router_data_dir,
            selected_data_path=output_path,
            data_dir=cfg.get("data_dir"),
            start_time=start_time,
            end_time=end_time,
        )
        
        log.info(f"质量选择的数据已保存到: {output_path}")
        log.info("=== 基于质量的数据选择完成 ===")
        
    except Exception as e:
        log.error(f"数据选择过程中发生错误: {e}")
        raise e


if __name__ == "__main__":
    main()
