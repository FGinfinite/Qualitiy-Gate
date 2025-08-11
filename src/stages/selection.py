# src/stages/selection.py
import json
import logging
import os
from typing import List, Tuple

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.clustering import ClusterBasedSelection
from src.data import load_local_datasets
from src.models.select_moe import SelectMoeForCausalLM, register_select_moe
from src.training.full_rank_finetuning import load_full_rank_weights


def get_model_and_tokenizer(cfg: DictConfig) -> Tuple[SelectMoeForCausalLM, AutoTokenizer]:
    """
    加载预训练的Select-MoE模型并应用全秩微调权重
    """
    # 注册Select-MoE模型
    register_select_moe()

    model_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }

    # 加载预转换的Select-MoE模型
    model = SelectMoeForCausalLM.from_pretrained(cfg.selector_model.path, **model_kwargs)

    # 加载全秩微调权重
    load_full_rank_weights(model, cfg.model_checkpoint_path)

    model.eval()

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(cfg.selector_model.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def calculate_quality_score_from_gates(
    quality_score_list: list,
    attention_mask: torch.Tensor,
    debug: bool = False,
) -> float:
    """
    使用质量门输出计算质量分数

    Args:
        quality_score_list: 各层质量门输出的分数列表
        attention_mask: 形状为 (seq_len,) 的注意力掩码
        debug: 是否打印调试信息

    Returns:
        质量分数：good_ratio的平均值
    """
    valid_mask = attention_mask.bool()
    actual_length = valid_mask.sum().item()

    if debug:
        log = logging.getLogger(__name__)
        log.debug(f"质量门分数数量: {len(quality_score_list)}")
        log.debug(f"实际token数量: {actual_length}")

    if actual_length == 0:
        return 0.0

    layer_quality_scores = []
    for layer_idx, quality_score in enumerate(quality_score_list):
        # quality_score: (seq_len, 1) -> 原始分数
        valid_quality_score = quality_score[valid_mask]  # (actual_length, 1)

        # 使用sigmoid转换为概率
        quality_probs = torch.sigmoid(valid_quality_score)  # (actual_length, 1)
        good_probs = quality_probs.squeeze(-1)  # (actual_length,)

        # 在该层内对所有token求平均
        layer_avg_good_prob = good_probs.mean().item()
        layer_quality_scores.append(layer_avg_good_prob)

        if debug and layer_idx < 3:
            log.debug(f"第{layer_idx + 1}层好数据概率: {layer_avg_good_prob:.6f}")

    # 对所有层求平均
    final_quality_score = sum(layer_quality_scores) / len(layer_quality_scores)

    if debug:
        log.debug(f"最终质量分数: {final_quality_score:.6f}")

    return final_quality_score


def cluster_based_selection(
    scored_data: List[dict],
    all_logits_by_dataset: dict,
    selection_percentage: float,
    clustering_method: str = "kmeans",
    clustering_params: dict = None,
    device: torch.device = None,
    debug_print: bool = False,
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
    cluster_selector = ClusterBasedSelection(device=device, debug_print=debug_print)

    # 执行聚类-轮选选择
    selected_data = cluster_selector.select_data_by_clustering(
        scored_data=scored_data,
        all_logits_by_dataset=all_logits_by_dataset,
        target_count=target_count,
        clustering_method=clustering_method,
        clustering_params=clustering_params or {},
    )

    return selected_data


def load_router_data(router_data_path: str) -> dict:
    """
    加载保存的完整路由数据

    Args:
        router_data_path: 路由数据文件路径

    Returns:
        包含完整路由信息的字典
    """
    router_data = torch.load(router_data_path, map_location="cpu")

    log = logging.getLogger(__name__)
    log.info(f"加载路由数据: {router_data_path}")
    log.info(f"  - 数据集: {router_data['dataset_name']}")
    log.info(f"  - 样本数: {router_data['num_samples']}")
    log.info(f"  - 质量门分数形状: {router_data['quality_score'].shape}")
    log.info(f"  - MoE路由logits形状: {router_data['moe_logits'].shape}")

    return router_data


def get_sample_router_info(router_data: dict, sample_id: str) -> dict:
    """
    根据样本ID获取对应的路由信息

    Args:
        router_data: 从load_router_data加载的数据字典
        sample_id: 样本的唯一ID（例如: "oasst1_25460"）

    Returns:
        包含该样本路由信息的字典
    """
    # 在样本ID列表中查找位置
    try:
        position = router_data["sample_ids"].index(sample_id)
    except ValueError:
        raise ValueError(f"样本ID '{sample_id}' 未在数据集 '{router_data['dataset_name']}' 中找到") from None

    return {
        "sample_id": sample_id,
        "dataset_name": router_data["dataset_name"],
        "quality_score": router_data["quality_score"][position],  # [num_layers, 1]
        "moe_logits": router_data["moe_logits"][position],  # [num_layers, num_experts]
        "position_in_dataset": position,
    }


def select(cfg: DictConfig) -> None:
    """
    数据选择阶段的主函数
    """
    log = logging.getLogger(__name__)
    log.info("--- 开始阶段2：数据选择 ---")

    # 确定目标设备
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

    # 如果使用GPU，清理缓存
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 1. 加载模型和分词器
    log.info(f"从检查点加载模型: {cfg.model_checkpoint_path}")
    model, tokenizer = get_model_and_tokenizer(cfg)
    num_hidden_layers = model.config.num_hidden_layers
    log.info(f"成功加载模型: {cfg.model_checkpoint_path}")

    # 2. 验证新架构
    with torch.no_grad():
        model_device = next(model.parameters()).device
        dummy_input = torch.ones(1, 10, dtype=torch.long, device=model_device)
        dummy_outputs = model(dummy_input, output_router_logits=True)

        # 验证输出格式
        if not isinstance(dummy_outputs.router_logits[0], dict):
            raise ValueError("模型不是新的两层路由架构，请使用正确的模型")

        if "quality_score" not in dummy_outputs.router_logits[0]:
            raise ValueError("模型缺少质量门输出，请使用正确的两层路由架构")

        if "moe_logits" not in dummy_outputs.router_logits[0]:
            raise ValueError("模型缺少MoE路由输出，请使用正确的两层路由架构")

    log.info("✓ 验证模型为新的两层路由架构")
    log.info(f"✓ 模型层数: {num_hidden_layers}")
    log.info(f"✓ 质量门输出维度: {dummy_outputs.router_logits[0]['quality_score'].shape[-1]}")
    log.info(f"✓ MoE专家数量: {dummy_outputs.router_logits[0]['moe_logits'].shape[-1]}")

    # 3. 加载和准备数据集
    log.info(f"加载数据集: {cfg.dataset.dataset_names}")
    dataset = load_local_datasets(
        data_dir=cfg.dataset.data_dir,
        dataset_names=cfg.dataset.dataset_names,
        sample_percentage=cfg.dataset.subset_ratio,
        seed=cfg.dataset.seed,
    )

    if cfg.dataset.shuffle:
        log.info("对数据集进行shuffle...")
        dataset = dataset.shuffle(seed=cfg.dataset.seed)

    log.info(f"总样本数: {len(dataset)}")
    log.info("✓ 数据集已准备完毕")

    # 4. 数据评分和路由logits记录
    scored_data = []
    all_router_data_by_dataset = {
        name: {
            "quality_score": [],  # 质量门分数
            "moe_logits": [],  # MoE路由logits
            "sample_ids": [],  # 样本ID
        }
        for name in cfg.dataset.dataset_names
    }
    dataset_sample_counts = {name: 0 for name in cfg.dataset.dataset_names}

    # 创建数据加载器
    def collate_fn(batch):
        return {
            "messages": [item["messages"] for item in batch],
            "dataset": [item["dataset"] for item in batch],
            "id": [item["id"] for item in batch],
        }

    dataloader = DataLoader(dataset, batch_size=cfg.data_process.batch_size, collate_fn=collate_fn)

    log.info("开始数据评分...")
    for i, batch in enumerate(tqdm(dataloader)):
        messages_list = batch["messages"]
        dataset_names = batch["dataset"]
        ids = batch["id"]

        # 将messages转换为文本
        texts = []
        for messages in messages_list:
            if isinstance(messages, list) and len(messages) > 0:
                text_parts = []
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    text_parts.append(f"{role}: {content}")
                texts.append("\n".join(text_parts))
            else:
                texts.append("")

        if i == 0:
            log.info("批次0，前3个文本示例:")
            for idx in range(min(3, len(texts))):
                log.info(f"{idx + 1}: {texts[idx][:100]}...")

        # 分词
        inputs = tokenizer(
            texts,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            max_length=cfg.dataset.max_sequence_length,
        )
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_router_logits=True)

        # 处理router logits
        batch_size = len(texts)
        sequence_length = inputs["input_ids"].shape[1]

        moe_router_logits = [layer_dict["moe_logits"] for layer_dict in outputs.router_logits]
        all_router_logits = torch.stack(moe_router_logits)

        if i == 0:
            log.info(f"Router logits形状: {all_router_logits.shape}")
            log.info(f"批次大小: {batch_size}, 序列长度: {sequence_length}")

        # 计算质量分数
        for j in range(batch_size):
            dataset_name = dataset_names[j]
            sample_attention_mask = inputs["attention_mask"][j]
            debug_mode = i == 0 and j < 3

            # 使用质量门输出计算质量分数
            quality_score_list = [layer_dict["quality_score"][j] for layer_dict in outputs.router_logits]
            quality_score = calculate_quality_score_from_gates(
                quality_score_list,
                sample_attention_mask,
                debug=debug_mode,
            )

            # 获取有效token的掩码信息
            valid_mask = sample_attention_mask.bool()
            actual_length = valid_mask.sum().item()

            # 记录完整的路由信息
            # 1. 收集质量门logits并在序列维度进行平均
            quality_score_averaged = []
            for layer_quality_score in quality_score_list:
                if actual_length > 0:
                    valid_quality_score = layer_quality_score[valid_mask]
                    layer_avg_quality_score = valid_quality_score.mean(dim=0)
                else:
                    layer_avg_quality_score = torch.zeros(1)
                quality_score_averaged.append(layer_avg_quality_score)

            quality_score_sample = torch.stack(quality_score_averaged)  # [num_layers, 1]

            # 2. 收集MoE路由logits并计算平均概率分布
            if actual_length > 0:
                moe_logits_list = []
                seq_len = inputs["input_ids"].shape[1]

                for _, layer_dict in enumerate(outputs.router_logits):
                    layer_moe_logits = layer_dict["moe_logits"]

                    # 提取第j个样本的logits
                    start_idx = j * seq_len
                    end_idx = (j + 1) * seq_len
                    sample_layer_logits = layer_moe_logits[start_idx:end_idx]

                    # 转换为概率分布
                    sample_layer_probs = torch.softmax(sample_layer_logits, dim=-1)

                    # 根据attention mask提取有效token的概率，然后求平均
                    valid_probs = sample_layer_probs[valid_mask]
                    if valid_probs.shape[0] > 0:
                        layer_avg_probs = valid_probs.mean(dim=0)
                    else:
                        num_experts = sample_layer_probs.shape[-1]
                        layer_avg_probs = torch.ones(num_experts) / num_experts

                    moe_logits_list.append(layer_avg_probs)

                sample_moe_matrix = torch.stack(moe_logits_list)  # [num_layers, num_experts]
            else:
                num_layers = len(outputs.router_logits)
                num_experts = outputs.router_logits[0]["moe_logits"].shape[-1]
                sample_moe_matrix = torch.ones(num_layers, num_experts) / num_experts
                quality_score_sample = torch.zeros(num_layers, 1)

            # 保存到数据结构中
            all_router_data_by_dataset[dataset_name]["quality_score"].append(quality_score_sample.cpu())
            all_router_data_by_dataset[dataset_name]["moe_logits"].append(sample_moe_matrix.cpu())
            all_router_data_by_dataset[dataset_name]["sample_ids"].append(ids[j])
            dataset_sample_counts[dataset_name] += 1

            if i == 0 and j < 3:
                current_sample_index = i * cfg.data_process.batch_size + j
                log.info(f"--- 样本 {current_sample_index} 详细计算过程 ---")
                log.info(f"  - 数据集: {dataset_names[j]}")
                log.info(f"  - 样本ID: {ids[j]}")
                log.info(f"  - 实际长度: {actual_length} / 总长度: {sample_attention_mask.shape[0]}")
                log.info(f"  - 最终质量分数: {quality_score:.6f}")

            scored_data.append(
                {
                    "dataset": dataset_names[j],
                    "id": ids[j],
                    "scores": quality_score,
                    "messages": messages_list[j],
                }
            )

    # 5. 立即保存完整的路由张量文件
    log.info("模型推理完成，立即保存完整路由张量文件...")
    output_path = hydra.utils.to_absolute_path(cfg.output_path)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    router_data_dir = os.path.join(output_dir, "router_data")
    os.makedirs(router_data_dir, exist_ok=True)

    for dataset_name in cfg.dataset.dataset_names:
        dataset_router_data = all_router_data_by_dataset[dataset_name]

        if dataset_router_data["quality_score"] and dataset_router_data["moe_logits"]:
            router_data_dict = {
                "quality_score": torch.stack(dataset_router_data["quality_score"]),  # [N, L, 1]
                "moe_logits": torch.stack(dataset_router_data["moe_logits"]),  # [N, L, E]
                "sample_ids": dataset_router_data["sample_ids"],
                "dataset_name": dataset_name,
                "num_samples": dataset_sample_counts[dataset_name],
                "metadata": {
                    "description": "完整的路由数据，包含质量门和MoE路由输出",
                    "quality_score_shape": "[N, num_layers, 1] - 质量门分数",
                    "moe_logits_shape": "[N, num_layers, num_experts] - MoE路由平均概率",
                    "sample_ids": "样本的唯一ID标识",
                },
            }

            router_data_path = os.path.join(router_data_dir, f"{dataset_name}_router_data.pt")
            torch.save(router_data_dict, router_data_path)

            log.info(f"数据集 '{dataset_name}' 的完整路由数据已保存到: {router_data_path}")
            log.info(f"  - 质量门分数形状: {router_data_dict['quality_score'].shape}")
            log.info(f"  - MoE路由logits形状: {router_data_dict['moe_logits'].shape}")
            log.info(f"  - 样本数: {router_data_dict['num_samples']}")

        else:
            log.warning(f"数据集 '{dataset_name}' 没有路由数据")

    # 释放模型和GPU内存
    log.info("释放模型实例和GPU内存...")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log.info("模型内存已释放")

    # 6. 检查是否跳过数据选择步骤
    skip_selection = getattr(cfg, "skip_data_selection", False)
    if skip_selection:
        log.info("根据配置，跳过数据选择步骤。")
        log.info("路由数据已保存，可以使用其他脚本进行数据选择算法实验。")
        log.info("--- 阶段2：数据选择完成（仅推理模式） ---")
        return

    # 7. 进行聚类数据选择
    log.info("开始聚类数据选择...")

    clustering_method = getattr(cfg, "clustering_method", "kmeans")
    clustering_params = getattr(cfg, "clustering_params", {})

    # 准备logits数据
    all_logits_by_dataset = {name: all_router_data_by_dataset[name]["moe_logits"] for name in cfg.dataset.dataset_names}

    try:
        selected_data = cluster_based_selection(
            scored_data=scored_data,
            all_logits_by_dataset=all_logits_by_dataset,
            selection_percentage=cfg.selection_percentage,
            clustering_method=clustering_method,
            clustering_params=clustering_params,
            device=device,
        )
    except torch.OutOfMemoryError as e:
        log.error(f"GPU内存不足，聚类计算失败: {e}")
        log.error("但是，宝贵的logits张量已经安全保存！")
        log.info("您可以使用独立脚本继续数据选择过程:")
        log.info(
            f"  python scripts/cluster_selection.py --router_data_dir {router_data_dir} "
            f"--output_path {output_path} --selection_percentage {cfg.selection_percentage}"
        )
        raise
    except Exception as e:
        log.error(f"聚类选择过程中发生错误: {e}")
        log.error("但是，宝贵的logits张量已经安全保存！")
        log.info("您可以使用独立脚本继续数据选择过程:")
        log.info(f"  python scripts/cluster_selection.py --router_data_dir {router_data_dir} --output_path {output_path}")
        raise

    log.info(f"选择了前 {len(selected_data)} 个样本 ({cfg.selection_percentage * 100:.2f}%)")
    if len(selected_data) > 3:
        log.info(f"前3个分数: {[d['scores'] for d in selected_data[:3]]}")
        log.info(f"后3个分数: {[d['scores'] for d in selected_data[-3:]]}")

    # 保存选择的数据
    with open(output_path, "w", encoding="utf-8") as f:
        for item in selected_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    log.info(f"筛选后的数据已保存到: {output_path}")
    log.info("--- 阶段2：数据选择完成 ---")
