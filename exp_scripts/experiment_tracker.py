#!/usr/bin/env python3
"""
Select-MoE实验进度跟踪脚本

该脚本分析阶段1-3的实验结果并显示它们的关系。
帮助识别哪些实验已完成并跟踪流水线进度。

使用方法:
    python experiment_tracker.py [--outputs-dir outputs] [--verbose] [--format table|tree] [--save-report]
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from prettytable import PrettyTable


@dataclass
class ExperimentInfo:
    """单个实验的信息"""

    stage: int
    path: Path
    name: str
    timestamp: str
    config: Dict
    status: str  # 'complete'完成, 'partial'部分完成, 'missing'缺失
    files: List[str]  # 找到的关键文件列表
    parent_path: Optional[str] = None  # 父实验路径
    evaluated_datasets: Optional[List[str]] = None  # 阶段4已完成评估的数据集列表


class ExperimentTracker:
    """实验进度跟踪主类"""

    def __init__(self, outputs_dir: str = "outputs"):
        self.outputs_dir = Path(outputs_dir)
        self.stage_1_exps: Dict[str, ExperimentInfo] = {}
        self.stage_2_exps: Dict[str, ExperimentInfo] = {}
        self.stage_3_exps: Dict[str, ExperimentInfo] = {}
        self.stage_4_exps: Dict[str, ExperimentInfo] = {}
        self.relationships: Dict[str, List[str]] = defaultdict(list)

    def check_stage_completion(self, exp_path: Path, stage: int) -> Tuple[str, List[str]]:
        """检查各阶段实验是否完成"""
        if stage == 4:
            return self.check_stage_4_completion(exp_path)

        required_files = {1: ["full_rank_weights.pt"], 2: ["router_data/", "selected_data.jsonl"], 3: ["adapter_model.safetensors"]}

        found_files = []
        for file in required_files[stage]:
            if file.endswith("/"):
                # 目录检查
                dir_path = exp_path / file[:-1]
                if dir_path.exists() and list(dir_path.glob("*.pt")):
                    found_files.append(file)
            else:
                # 文件检查
                if (exp_path / file).exists():
                    found_files.append(file)

        required_count = len(required_files[stage])
        if len(found_files) == required_count:
            return "complete", found_files
        elif len(found_files) > 0:
            return "partial", found_files
        else:
            return "missing", found_files

    def load_config(self, exp_path: Path) -> Dict:
        """从.hydra/config.yaml加载实验配置"""
        config_path = exp_path / ".hydra" / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"警告: 无法加载配置文件 {config_path}: {e}")
        return {}

    def check_stage_4_completion(self, exp_path: Path) -> Tuple[str, List[str]]:
        """检查阶段4实验的评估完成状态"""
        json_files = list(exp_path.glob("*.json"))
        if not json_files:
            return "missing", []

        evaluated_datasets = set()
        found_files = []

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    groups = data.get("groups", {})

                    # 查找主数据集（无下划线的键值）
                    for dataset_name in groups.keys():
                        if "_" not in dataset_name:
                            evaluated_datasets.add(dataset_name)

                    found_files.append(json_file.name)

            except Exception as e:
                print(f"警告: 无法解析评估结果文件 {json_file}: {e}")

        # 检查是否包含必需的数据集 bbh 和 mmlu
        has_bbh = "bbh" in evaluated_datasets
        has_mmlu = "mmlu" in evaluated_datasets

        if has_bbh and has_mmlu:
            return "complete", found_files
        elif has_bbh or has_mmlu:
            return "partial", found_files
        else:
            return "missing", found_files

    def parse_stage_4_path(self, stage4_dir_name: str) -> Optional[str]:
        """解析阶段4目录名，返回对应的阶段3路径"""
        # 格式: outputs__stage_3_finetune__时间戳__实验名称__
        if not stage4_dir_name.startswith("outputs__stage_3_finetune__"):
            return None

        # 移除前缀，但暂时不移除后缀
        path_part = stage4_dir_name[len("outputs__stage_3_finetune__") :]

        # 查找第一个日期模式 (YYYY-MM-DD)
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", path_part)
        if not date_match:
            return None

        date_part = date_match.group(1)

        # 获取日期后面的部分作为实验名称
        date_end_pos = date_match.end()
        if date_end_pos < len(path_part) and path_part[date_end_pos : date_end_pos + 2] == "__":
            exp_name_encoded = path_part[date_end_pos + 2 :]

            # 先移除可能的末尾双下划线
            if exp_name_encoded.endswith("__"):
                exp_name_encoded = exp_name_encoded[:-2]

            # 恢复阶段3的实验名称格式
            # 输入: 09-48-18-MODEL=__batch=128_lr=2e-05_tag=SE_finetune__-DATA=__19-57-09-batch=16_lr=0.0001_loss=sigmoid_lossWeight=1_sampleWise=True_tag=none
            # 输出: 09-48-18-MODEL=<|batch=128_lr=2e-05_tag=SE_finetune|>-DATA=<|19-57-09-batch=16_lr=0.0001_loss=sigmoid_lossWeight=1_sampleWise=True_tag=none|>

            exp_name = exp_name_encoded
            # 使用循环替换所有 =__...__ 模式，处理末尾没有 __ 的情况
            while True:
                # 找到 =__ 的位置
                start_pos = exp_name.find("=__")
                if start_pos == -1:
                    break

                # 找到匹配的 __ 的位置，如果找不到，说明是最后一段
                end_pos = exp_name.find("__", start_pos + 3)
                if end_pos == -1:
                    # 处理最后一段（没有结束的 __）
                    before = exp_name[: start_pos + 1]  # 包含 =
                    middle = exp_name[start_pos + 3 :]  # 到末尾
                    exp_name = before + "<|" + middle + "|>"
                    break
                else:
                    # 处理中间段（有结束的 __）
                    before = exp_name[: start_pos + 1]  # 包含 =
                    middle = exp_name[start_pos + 3 : end_pos]  # 不包含 __
                    after = exp_name[end_pos + 2 :]  # 不包含 __
                    exp_name = before + "<|" + middle + "|>" + after

            # 构建阶段3路径
            stage3_path = f"outputs/stage_3_finetune/{date_part}/{exp_name}"
            return stage3_path

        return None

    def extract_timestamp(self, exp_name: str) -> str:
        """从实验名称中提取时间戳"""
        parts = exp_name.split("-")
        if len(parts) >= 3:
            return f"{parts[0]}-{parts[1]}-{parts[2]}"
        return "未知"

    def scan_experiments(self):
        """扫描所有实验目录并收集信息"""
        stage_configs = [(1, "stage_1_pretrain"), (2, "stage_2_selection"), (3, "stage_3_finetune")]

        # 扫描阶段1-3
        for stage, stage_dir_name in stage_configs:
            stage_dir = self.outputs_dir / stage_dir_name
            if not stage_dir.exists():
                continue

            exp_dict = getattr(self, f"stage_{stage}_exps")

            for date_dir in stage_dir.iterdir():
                if date_dir.is_dir() and date_dir.name != "multirun.yaml":
                    for exp_dir in date_dir.iterdir():
                        if exp_dir.is_dir():
                            status, files = self.check_stage_completion(exp_dir, stage)
                            config = self.load_config(exp_dir)
                            timestamp = self.extract_timestamp(exp_dir.name)

                            # 提取父路径
                            parent_path = None
                            if stage == 2 and "model_checkpoint_path" in config:
                                parent_path = config["model_checkpoint_path"]
                            elif stage == 3 and config.get("dataset", {}).get("mode") == "subset":
                                if "data_path" in config["dataset"]:
                                    parent_path = str(Path(config["dataset"]["data_path"]).parent)

                            exp_info = ExperimentInfo(
                                stage=stage,
                                path=exp_dir,
                                name=exp_dir.name,
                                timestamp=timestamp,
                                config=config,
                                status=status,
                                files=files,
                                parent_path=parent_path,
                            )
                            exp_dict[str(exp_dir)] = exp_info

        # 扫描阶段4
        self.scan_stage_4_experiments()

    def scan_stage_4_experiments(self):
        """扫描阶段4评估实验"""
        stage4_dir = self.outputs_dir / "stage_4_eval"
        if not stage4_dir.exists():
            return

        for eval_dir in stage4_dir.iterdir():
            if eval_dir.is_dir():
                # 检查评估完成状态
                status, files = self.check_stage_completion(eval_dir, 4)

                # 解析阶段4路径获取对应的阶段3路径
                parent_path = self.parse_stage_4_path(eval_dir.name)

                # 提取时间戳（从目录名中提取）
                timestamp = self.extract_stage_4_timestamp(eval_dir.name)

                # 获取已评估的数据集列表
                evaluated_datasets = self.get_evaluated_datasets(eval_dir)

                exp_info = ExperimentInfo(
                    stage=4,
                    path=eval_dir,
                    name=eval_dir.name,
                    timestamp=timestamp,
                    config={},  # 阶段4没有配置文件
                    status=status,
                    files=files,
                    parent_path=parent_path,
                    evaluated_datasets=evaluated_datasets,
                )
                self.stage_4_exps[str(eval_dir)] = exp_info

    def extract_stage_4_timestamp(self, stage4_dir_name: str) -> str:
        """从阶段4目录名中提取时间戳"""
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", stage4_dir_name)
        if date_match:
            return date_match.group(1)
        return "未知"

    def get_evaluated_datasets(self, exp_path: Path) -> List[str]:
        """获取已完成评估的数据集列表"""
        json_files = list(exp_path.glob("*.json"))
        evaluated_datasets = set()

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    groups = data.get("groups", {})

                    # 查找主数据集（无下划线的键值）
                    for dataset_name in groups.keys():
                        if "_" not in dataset_name:
                            evaluated_datasets.add(dataset_name)

            except Exception:
                continue

        return sorted(list(evaluated_datasets))

    def build_relationships(self):
        """建立实验之间的关系映射"""
        # 建立阶段1 -> 阶段2关系
        for stage2_path, stage2_exp in self.stage_2_exps.items():
            if stage2_exp.parent_path:
                parent_abs_path = str(Path(stage2_exp.parent_path).resolve())
                for stage1_path, stage1_exp in self.stage_1_exps.items():
                    if str(stage1_exp.path.resolve()) == parent_abs_path:
                        self.relationships[stage1_path].append(stage2_path)
                        break

        # 建立阶段2 -> 阶段3关系
        for stage3_path, stage3_exp in self.stage_3_exps.items():
            if stage3_exp.parent_path:
                parent_abs_path = str(Path(stage3_exp.parent_path).resolve())
                for stage2_path, stage2_exp in self.stage_2_exps.items():
                    if str(stage2_exp.path.resolve()) == parent_abs_path:
                        self.relationships[stage2_path].append(stage3_path)
                        break

        # 建立阶段3 -> 阶段4关系
        for stage4_path, stage4_exp in self.stage_4_exps.items():
            if stage4_exp.parent_path:
                # 阶段4的parent_path是解析出来的阶段3路径（相对路径格式）
                expected_stage3_path = stage4_exp.parent_path

                # 查找匹配的阶段3实验
                for stage3_path, stage3_exp in self.stage_3_exps.items():
                    # 构建阶段3的相对路径格式进行匹配
                    stage3_relative_path = f"outputs/stage_3_finetune/{stage3_exp.path.parent.name}/{stage3_exp.path.name}"

                    if stage3_relative_path == expected_stage3_path:
                        self.relationships[stage3_path].append(stage4_path)
                        break

    def get_status_icon(self, status: str) -> str:
        """获取状态图标"""
        return {"complete": "✅", "partial": "⚠️", "missing": "❌"}.get(status, "❓")

    def get_status_text(self, status: str) -> str:
        """获取状态中文描述"""
        return {"complete": "完成", "partial": "部分完成", "missing": "缺失"}.get(status, "未知")

    def create_summary_table(self) -> PrettyTable:
        """创建汇总表格"""
        table = PrettyTable()
        table.field_names = ["阶段", "完成数量", "总数量", "完成率", "状态"]

        stages_data = [
            ("阶段1-路由预训练", self.stage_1_exps),
            ("阶段2-数据选择", self.stage_2_exps),
            ("阶段3-模型微调", self.stage_3_exps),
            ("阶段4-模型评估", self.stage_4_exps),
        ]

        for stage_name, experiments in stages_data:
            complete_count = sum(1 for exp in experiments.values() if exp.status == "complete")
            total_count = len(experiments)
            percentage = f"({complete_count / total_count * 100:.1f}%)" if total_count > 0 else "(0.0%)"

            if complete_count == total_count and total_count > 0:
                status = "✅ 全部完成"
            elif complete_count > 0:
                status = "⚠️ 部分完成"
            else:
                status = "❌ 尚未开始"

            table.add_row([stage_name, complete_count, total_count, percentage, status])

        return table

    def create_pipeline_table(self) -> PrettyTable:
        """创建流水线表格，支持一对多关系"""
        table = PrettyTable()
        table.field_names = ["层级", "阶段1实验", "状态1", "阶段2实验", "状态2", "阶段3实验", "状态3", "阶段4实验", "状态4"]
        # 移除max_width限制，显示完整实验名称

        for stage1_path, stage1_exp in sorted(self.stage_1_exps.items(), key=lambda x: x[1].path.name):
            stage1_name = stage1_exp.name  # 显示完整名称
            stage1_status = self.get_status_icon(stage1_exp.status)

            stage2_children = self.relationships.get(stage1_path, [])

            if not stage2_children:
                table.add_row(["1", stage1_name, stage1_status, "-", "-", "-", "-", "-", "-"])
            else:
                for i, stage2_path in enumerate(sorted(stage2_children)):
                    stage2_exp = self.stage_2_exps.get(stage2_path)
                    if not stage2_exp:
                        continue

                    stage2_name = stage2_exp.name  # 显示完整名称
                    stage2_status = self.get_status_icon(stage2_exp.status)

                    stage3_children = self.relationships.get(stage2_path, [])

                    if not stage3_children:
                        level = "1" if i == 0 else "1.1"
                        s1_name = stage1_name if i == 0 else "↳"
                        s1_status = stage1_status if i == 0 else "│"
                        table.add_row([level, s1_name, s1_status, stage2_name, stage2_status, "-", "-", "-", "-"])
                    else:
                        for j, stage3_path in enumerate(sorted(stage3_children)):
                            stage3_exp = self.stage_3_exps.get(stage3_path)
                            if not stage3_exp:
                                continue

                            stage3_name = stage3_exp.name  # 显示完整名称
                            stage3_status = self.get_status_icon(stage3_exp.status)

                            stage4_children = self.relationships.get(stage3_path, [])

                            if not stage4_children:
                                if i == 0 and j == 0:
                                    level = "1"
                                    s1_name, s1_status = stage1_name, stage1_status
                                    s2_name, s2_status = stage2_name, stage2_status
                                elif j == 0:
                                    level = "1.1"
                                    s1_name, s1_status = "↳", "│"
                                    s2_name, s2_status = stage2_name, stage2_status
                                else:
                                    level = "1.1.1"
                                    s1_name, s1_status = "↳", "│"
                                    s2_name, s2_status = "↳", "│"

                                table.add_row([level, s1_name, s1_status, s2_name, s2_status, stage3_name, stage3_status, "-", "-"])
                            else:
                                for k, stage4_path in enumerate(sorted(stage4_children)):
                                    stage4_exp = self.stage_4_exps.get(stage4_path)
                                    if not stage4_exp:
                                        continue

                                    stage4_name = stage4_exp.name  # 显示完整名称
                                    stage4_status = self.get_status_icon(stage4_exp.status)

                                    if i == 0 and j == 0 and k == 0:
                                        level = "1"
                                        s1_name, s1_status = stage1_name, stage1_status
                                        s2_name, s2_status = stage2_name, stage2_status
                                        s3_name, s3_status = stage3_name, stage3_status
                                    elif j == 0 and k == 0:
                                        level = "1.1"
                                        s1_name, s1_status = "↳", "│"
                                        s2_name, s2_status = stage2_name, stage2_status
                                        s3_name, s3_status = stage3_name, stage3_status
                                    elif k == 0:
                                        level = "1.1.1"
                                        s1_name, s1_status = "↳", "│"
                                        s2_name, s2_status = "↳", "│"
                                        s3_name, s3_status = stage3_name, stage3_status
                                    else:
                                        level = "1.1.1.1"
                                        s1_name, s1_status = "↳", "│"
                                        s2_name, s2_status = "↳", "│"
                                        s3_name, s3_status = "↳", "│"

                                    table.add_row([level, s1_name, s1_status, s2_name, s2_status, s3_name, s3_status, stage4_name, stage4_status])

        return table

    def escape_markdown_pipes(self, text: str) -> str:
        """转义Markdown表格中的管道字符"""
        return str(text).replace("|", "&#124;")

    def generate_markdown_report(self, output_path: Path) -> str:
        """生成Markdown报告"""
        lines = [
            "# Select-MoE实验进度报告",
            "",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"数据源: {self.outputs_dir}",
            "",
            "## 概览汇总",
            "",
        ]

        # 汇总表格
        summary_table = self.create_summary_table()
        lines.extend(["| 阶段 | 完成数量 | 总数量 | 完成率 | 状态 |", "|------|----------|--------|--------|------|"])

        for row in summary_table._rows:
            escaped_row = [self.escape_markdown_pipes(cell) for cell in row]
            lines.append(f"| {' | '.join(escaped_row)} |")

        lines.extend(["", "## 实验流水线详情", "", "### 完整流水线表格", ""])

        # 流水线表格
        pipeline_table = self.create_pipeline_table()
        lines.extend(
            [
                "| 层级 | 阶段1实验 | 状态1 | 阶段2实验 | 状态2 | 阶段3实验 | 状态3 | 阶段4实验 | 状态4 |",
                "|------|-----------|-------|-----------|-------|-----------|-------|-----------|-------|",
            ]
        )

        for row in pipeline_table._rows:
            escaped_row = [self.escape_markdown_pipes(cell) for cell in row]
            lines.append(f"| {' | '.join(escaped_row)} |")

        # 孤儿实验
        orphaned_stage2 = [exp for path, exp in self.stage_2_exps.items() if not any(path in children for children in self.relationships.values())]
        orphaned_stage3 = [exp for path, exp in self.stage_3_exps.items() if not any(path in children for children in self.relationships.values())]
        orphaned_stage4 = [exp for path, exp in self.stage_4_exps.items() if not any(path in children for children in self.relationships.values())]

        if orphaned_stage2 or orphaned_stage3 or orphaned_stage4:
            lines.extend(["", "## 孤儿实验（无父实验关系）", ""])

            if orphaned_stage2:
                lines.extend(["### 孤儿阶段2实验", "", "| 实验名称 | 状态 | 期望父实验 |", "|----------|------|------------|"])
                for exp in sorted(orphaned_stage2, key=lambda x: x.name):
                    icon = self.get_status_icon(exp.status)
                    name = self.escape_markdown_pipes(exp.name)
                    parent = self.escape_markdown_pipes(exp.parent_path or "未知")
                    lines.append(f"| {name} | {icon} | {parent} |")
                lines.append("")

            if orphaned_stage3:
                lines.extend(["### 孤儿阶段3实验", "", "| 实验名称 | 状态 | 期望父实验 |", "|----------|------|------------|"])
                for exp in sorted(orphaned_stage3, key=lambda x: x.name):
                    icon = self.get_status_icon(exp.status)
                    name = self.escape_markdown_pipes(exp.name)
                    parent = self.escape_markdown_pipes(exp.parent_path or "无")
                    lines.append(f"| {name} | {icon} | {parent} |")

            if orphaned_stage4:
                lines.extend(["### 孤儿阶段4实验", "", "| 实验名称 | 状态 | 期望父实验 | 已评估数据集 |", "|----------|------|------------|--------------|"])
                for exp in sorted(orphaned_stage4, key=lambda x: x.name):
                    icon = self.get_status_icon(exp.status)
                    name = self.escape_markdown_pipes(exp.name)
                    parent = self.escape_markdown_pipes(exp.parent_path or "无")
                    datasets = ", ".join(exp.evaluated_datasets) if exp.evaluated_datasets else "无"
                    datasets = self.escape_markdown_pipes(datasets)
                    lines.append(f"| {name} | {icon} | {parent} | {datasets} |")

        lines.extend(
            [
                "",
                "---",
                "",
                f"**报告生成于**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
                f"**总计**: 阶段1({len(self.stage_1_exps)}) | 阶段2({len(self.stage_2_exps)}) | 阶段3({len(self.stage_3_exps)}) | 阶段4({len(self.stage_4_exps)})",
            ]
        )

        content = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return content

    def print_results(self, verbose: bool = False, output_format: str = "tree", save_report: bool = False):
        """打印结果"""
        print("\n" + "=" * 80)
        print("SELECT-MOE实验进度跟踪 - 阶段1-4")
        print("=" * 80)

        # 统计数据
        stage1_complete = sum(1 for exp in self.stage_1_exps.values() if exp.status == "complete")
        stage2_complete = sum(1 for exp in self.stage_2_exps.values() if exp.status == "complete")
        stage3_complete = sum(1 for exp in self.stage_3_exps.values() if exp.status == "complete")
        stage4_complete = sum(1 for exp in self.stage_4_exps.values() if exp.status == "complete")

        print("\n汇总:")
        print(f"  阶段1-路由预训练: {stage1_complete}/{len(self.stage_1_exps)} 完成")
        print(f"  阶段2-数据选择: {stage2_complete}/{len(self.stage_2_exps)} 完成")
        print(f"  阶段3-模型微调: {stage3_complete}/{len(self.stage_3_exps)} 完成")
        print(f"  阶段4-模型评估: {stage4_complete}/{len(self.stage_4_exps)} 完成")
        print("\n状态说明: ✅ 完成 | ⚠️ 部分完成 | ❌ 缺失")

        if output_format == "table":
            print("-" * 80)
            print("\n阶段汇总:")
            print(self.create_summary_table())
            print("\n实验流水线:")
            print(self.create_pipeline_table())
            self._print_orphaned_experiments(verbose)
        else:
            self._print_tree_format(verbose)

        if save_report:
            report_path = self.outputs_dir / f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            self.generate_markdown_report(report_path)
            print(f"\n报告已保存到: {report_path}")

        print(f"\n{'=' * 80}")

    def _print_tree_format(self, verbose: bool = False):
        """打印树形格式"""
        print("-" * 80)

        for stage1_path, stage1_exp in sorted(self.stage_1_exps.items(), key=lambda x: x[1].path.name):
            icon = self.get_status_icon(stage1_exp.status)
            print(f"\n{icon} 阶段1: {stage1_exp.name}")

            if verbose:
                print(f"    路径: {stage1_exp.path}")
                print(f"    文件: {', '.join(stage1_exp.files) if stage1_exp.files else '无'}")

            stage2_children = self.relationships.get(stage1_path, [])
            for i, stage2_path in enumerate(sorted(stage2_children)):
                stage2_exp = self.stage_2_exps.get(stage2_path)
                if stage2_exp:
                    is_last_stage2 = i == len(stage2_children) - 1
                    prefix = "└──" if is_last_stage2 else "├──"
                    icon = self.get_status_icon(stage2_exp.status)
                    print(f"  {prefix} {icon} 阶段2: {stage2_exp.name}")

                    if verbose:
                        indent = "      " if is_last_stage2 else "  │   "
                        print(f"{indent}路径: {stage2_exp.path}")
                        print(f"{indent}文件: {', '.join(stage2_exp.files) if stage2_exp.files else '无'}")

                    stage3_children = self.relationships.get(stage2_path, [])
                    for j, stage3_path in enumerate(sorted(stage3_children)):
                        stage3_exp = self.stage_3_exps.get(stage3_path)
                        if stage3_exp:
                            is_last_stage3 = j == len(stage3_children) - 1
                            prefix = ("    └──" if is_last_stage3 else "    ├──") if is_last_stage2 else ("  │ └──" if is_last_stage3 else "  │ ├──")

                            icon = self.get_status_icon(stage3_exp.status)
                            print(f"{prefix} {icon} 阶段3: {stage3_exp.name}")

                            if verbose:
                                indent = "        " if is_last_stage2 else "  │     "
                                print(f"{indent}路径: {stage3_exp.path}")
                                print(f"{indent}文件: {', '.join(stage3_exp.files) if stage3_exp.files else '无'}")

                            # 添加阶段4
                            stage4_children = self.relationships.get(stage3_path, [])
                            for k, stage4_path in enumerate(sorted(stage4_children)):
                                stage4_exp = self.stage_4_exps.get(stage4_path)
                                if stage4_exp:
                                    is_last_stage4 = k == len(stage4_children) - 1

                                    # 根据上级状态构建前缀
                                    if is_last_stage2 and is_last_stage3:
                                        stage4_prefix = "      └──" if is_last_stage4 else "      ├──"
                                    elif is_last_stage2:
                                        stage4_prefix = "    │ └──" if is_last_stage4 else "    │ ├──"
                                    elif is_last_stage3:
                                        stage4_prefix = "  │   └──" if is_last_stage4 else "  │   ├──"
                                    else:
                                        stage4_prefix = "  │ │ └──" if is_last_stage4 else "  │ │ ├──"

                                    icon = self.get_status_icon(stage4_exp.status)
                                    print(f"{stage4_prefix} {icon} 阶段4: {stage4_exp.name}")

                                    if verbose:
                                        if is_last_stage2 and is_last_stage3:
                                            indent = "          " if is_last_stage4 else "      │   "
                                        elif is_last_stage2:
                                            indent = "    │     " if is_last_stage4 else "    │ │   "
                                        elif is_last_stage3:
                                            indent = "  │       " if is_last_stage4 else "  │   │   "
                                        else:
                                            indent = "  │ │     " if is_last_stage4 else "  │ │ │   "

                                        print(f"{indent}路径: {stage4_exp.path}")
                                        print(f"{indent}文件: {', '.join(stage4_exp.files) if stage4_exp.files else '无'}")
                                        if stage4_exp.evaluated_datasets:
                                            datasets = ", ".join(stage4_exp.evaluated_datasets)
                                            print(f"{indent}已评估: {datasets}")

        self._print_orphaned_experiments(verbose)

    def _print_orphaned_experiments(self, verbose: bool = False):
        """打印孤儿实验"""
        orphaned_stage2 = [exp for path, exp in self.stage_2_exps.items() if not any(path in children for children in self.relationships.values())]
        orphaned_stage3 = [exp for path, exp in self.stage_3_exps.items() if not any(path in children for children in self.relationships.values())]
        orphaned_stage4 = [exp for path, exp in self.stage_4_exps.items() if not any(path in children for children in self.relationships.values())]

        if orphaned_stage2 or orphaned_stage3 or orphaned_stage4:
            print(f"\n{'=' * 40}")
            print("孤儿实验（无父实验）")
            print(f"{'=' * 40}")

            if orphaned_stage2:
                print("\n孤儿阶段2实验:")
                for exp in sorted(orphaned_stage2, key=lambda x: x.name):
                    icon = self.get_status_icon(exp.status)
                    print(f"  {icon} {exp.name}")
                    if verbose:
                        print(f"      期望的父实验: {exp.parent_path}")

            if orphaned_stage3:
                print("\n孤儿阶段3实验:")
                for exp in sorted(orphaned_stage3, key=lambda x: x.name):
                    icon = self.get_status_icon(exp.status)
                    print(f"  {icon} {exp.name}")
                    if verbose:
                        print(f"      期望的父实验: {exp.parent_path}")

            if orphaned_stage4:
                print("\n孤儿阶段4实验:")
                for exp in sorted(orphaned_stage4, key=lambda x: x.name):
                    icon = self.get_status_icon(exp.status)
                    print(f"  {icon} {exp.name}")
                    if verbose:
                        print(f"      期望的父实验: {exp.parent_path}")
                        if exp.evaluated_datasets:
                            datasets = ", ".join(exp.evaluated_datasets)
                            print(f"      已评估数据集: {datasets}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="跟踪Select-MoE实验进度")
    parser.add_argument("--outputs-dir", default="outputs", help="输出目录路径 (默认: outputs)")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    parser.add_argument("--format", choices=["tree", "table"], default="tree", help="输出格式: tree(树形) 或 table(表格) (默认: tree)")
    parser.add_argument("--save-report", action="store_true", help="在outputs目录中生成Markdown报告")

    args = parser.parse_args()

    tracker = ExperimentTracker(args.outputs_dir)

    print("正在扫描实验...")
    tracker.scan_experiments()

    print("正在建立关系...")
    tracker.build_relationships()

    tracker.print_results(verbose=args.verbose, output_format=args.format, save_report=args.save_report)


if __name__ == "__main__":
    main()
