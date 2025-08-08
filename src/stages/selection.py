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
        "device_map": "auto",  # 自动设备映射，有助于内存优化
    }

    # 加载预转换的Select-MoE模型
    model = SelectMoeForCausalLM.from_pretrained(cfg.selector_model.path, **model_kwargs)

    # 加载全秩微调权重
    load_full_rank_weights(model, cfg.model_checkpoint_path)

    # device_map="auto" 已自动处理设备分配
    model.eval()

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(cfg.selector_model.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def calculate_quality_score_from_gates(
    quality_logits_list: list,
    attention_mask: torch.Tensor,
    debug: bool = False,
) -> float:
    """
    使用新架构中的质量门输出计算质量分数

    Args:
        quality_logits_list: 各层质量门输出的logits列表
        attention_mask: 形状为 (seq_len,) 的注意力掩码
        debug: 是否打印调试信息

    Returns:
        质量分数：good_ratio的平均值
    """
    valid_mask = attention_mask.bool()
    actual_length = valid_mask.sum().item()

    if debug:
        log = logging.getLogger(__name__)
        log.debug(f"    [函数内部] 质量门logits数量: {len(quality_logits_list)}")
        log.debug(f"    [函数内部] 实际token数量: {actual_length}")

    if actual_length == 0:
        return 0.0

    layer_quality_scores = []
    for layer_idx, quality_logits in enumerate(quality_logits_list):
        # quality_logits: (seq_len, 2) -> [good_prob, bad_prob]
        valid_quality_logits = quality_logits[valid_mask]  # (actual_length, 2)

        # 对每个token应用softmax得到概率
        quality_probs = torch.softmax(valid_quality_logits, dim=-1)  # (actual_length, 2)
        good_probs = quality_probs[:, 0]  # (actual_length,)

        # 在该层内对所有token求平均
        layer_avg_good_prob = good_probs.mean().item()
        layer_quality_scores.append(layer_avg_good_prob)

        if debug and layer_idx < 3:
            log.debug(f"    [函数内部] 第{layer_idx + 1}层好数据概率: {layer_avg_good_prob:.6f}")

    # 对所有层求平均
    final_quality_score = sum(layer_quality_scores) / len(layer_quality_scores)

    if debug:
        log.debug(f"    [函数内部] 最终质量分数: {final_quality_score:.6f}")

    return final_quality_score


def compute_wasserstein_distance_matrix(
    logits_tensors: List[torch.Tensor], device: torch.device, batch_size: int = 1000
) -> torch.Tensor:
    """
    计算所有样本对之间的Wasserstein距离矩阵（GPU加速）

    Args:
        logits_tensors: 列表，每个元素为 [L, E] 形状的张量
        device: GPU设备
        batch_size: 批处理大小，控制GPU内存使用

    Returns:
        距离矩阵: [N, N] 形状的GPU张量
    """
    n_samples = len(logits_tensors)
    log = logging.getLogger(__name__)
    log.info(f"使用GPU加速计算 {n_samples} 个样本的Wasserstein距离矩阵...")

    # 将所有logits转移到GPU并转换为概率分布
    prob_tensors = []
    for logits in logits_tensors:
        # 转移到GPU并计算概率分布
        probs = torch.softmax(logits.float().to(device), dim=-1)  # [L, E]
        prob_tensors.append(probs)

    # 堆叠成一个大张量 [N, L, E]
    all_probs = torch.stack(prob_tensors)  # [N, L, E]
    n_samples = all_probs.shape[0]

    log.info(f"概率张量形状: {all_probs.shape}")

    # 初始化距离矩阵
    distance_matrix = torch.zeros(n_samples, n_samples, device=device)

    # 分批处理以节省GPU内存
    for start_i in tqdm(range(0, n_samples, batch_size), desc="GPU距离矩阵计算"):
        end_i = min(start_i + batch_size, n_samples)
        batch_probs_i = all_probs[start_i:end_i]  # [batch_i, L, E]

        for start_j in range(start_i, n_samples, batch_size):
            end_j = min(start_j + batch_size, n_samples)
            batch_probs_j = all_probs[start_j:end_j]  # [batch_j, L, E]

            # 计算这个批次的距离
            batch_distances = compute_batch_wasserstein_distance_gpu(batch_probs_i, batch_probs_j)

            # 填充距离矩阵
            distance_matrix[start_i:end_i, start_j:end_j] = batch_distances

            # 对称填充（如果不是对角块）
            if start_i != start_j:
                distance_matrix[start_j:end_j, start_i:end_i] = batch_distances.T

    log.info(f"GPU距离矩阵计算完成，形状: {distance_matrix.shape}")
    return distance_matrix


def compute_batch_wasserstein_distance_gpu(batch_probs_i: torch.Tensor, batch_probs_j: torch.Tensor) -> torch.Tensor:
    """
    GPU上计算批次间的Wasserstein距离

    Args:
        batch_probs_i: [batch_i, L, E] 第一组样本的概率分布
        batch_probs_j: [batch_j, L, E] 第二组样本的概率分布

    Returns:
        距离矩阵: [batch_i, batch_j]
    """
    # batch_i, n_layers, n_experts = batch_probs_i.shape  # 未使用，注释掉

    # 扩展维度用于广播计算
    probs_i_expanded = batch_probs_i.unsqueeze(1)  # [batch_i, 1, L, E]
    probs_j_expanded = batch_probs_j.unsqueeze(0)  # [1, batch_j, L, E]

    # 计算每层的累积分布函数 (CDF)
    cdf_i = torch.cumsum(probs_i_expanded, dim=-1)  # [batch_i, 1, L, E]
    cdf_j = torch.cumsum(probs_j_expanded, dim=-1)  # [1, batch_j, L, E]

    # 计算Wasserstein距离 = L1距离的CDF差值
    # 对于离散分布，Wasserstein距离 = sum(|CDF_i - CDF_j|)
    cdf_diff = torch.abs(cdf_i - cdf_j)  # [batch_i, batch_j, L, E]

    # 对expert维度求和（每层的Wasserstein距离）
    layer_distances = torch.sum(cdf_diff, dim=-1)  # [batch_i, batch_j, L]

    # 对layer维度求和（总Wasserstein距离）
    total_distances = torch.sum(layer_distances, dim=-1)  # [batch_i, batch_j]

    return total_distances




def farthest_point_sampling_gpu(
    distance_matrix: torch.Tensor, n_samples: int, quality_scores: List[float] = None, seed: int = 42, log_interval: int = 100
) -> List[int]:
    """
    使用GPU加速的最远点采样(FPS)算法选择多样化样本

    Args:
        distance_matrix: [N, N] GPU上的距离矩阵
        n_samples: 要选择的样本数量
        quality_scores: 质量分数列表，用于选择初始点（可选）
        seed: 随机种子
        log_interval: 日志输出间隔

    Returns:
        选中样本的索引列表
    """
    log = logging.getLogger(__name__)
    device = distance_matrix.device
    n_total = distance_matrix.shape[0]

    if n_samples >= n_total:
        return list(range(n_total))

    # 设置随机种子
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    # 初始化选择状态和距离向量
    selected = torch.zeros(n_total, dtype=torch.bool, device=device)
    min_distances = torch.full((n_total,), float('inf'), device=device)
    selected_indices = []

    log.info(f"GPU FPS: 从 {n_total} 个样本中选择 {n_samples} 个")

    # 1. 选择初始点 - 优先使用质量分数最高的点
    if quality_scores is not None and len(quality_scores) == n_total:
        # 使用质量分数最高的点作为初始点
        quality_tensor = torch.tensor(quality_scores, device=device)
        first_idx = quality_tensor.argmax().item()
        log.info(f"FPS初始点: {first_idx} (质量分数: {quality_scores[first_idx]:.6f})")
    else:
        # 回退到随机选择
        first_idx = torch.randint(0, n_total, (1,), device=device).item()
        log.info(f"FPS初始点: {first_idx} (随机选择)")
    
    selected[first_idx] = True
    selected_indices.append(first_idx)

    # 更新到初始点的距离
    min_distances = distance_matrix[first_idx].clone()
    min_distances[first_idx] = 0  # 已选点距离设为0

    # 2. 贪心选择剩余点
    for step in range(1, n_samples):
        # 在未选择的点中找到距离最大的点（向量化操作）
        candidate_distances = min_distances.clone()
        candidate_distances[selected] = -1  # 已选点设为负值，排除

        next_idx = candidate_distances.argmax().item()
        max_distance = candidate_distances[next_idx].item()

        # 更新选择状态
        selected[next_idx] = True
        selected_indices.append(next_idx)

        # 批量更新所有点到已选点集的最小距离（向量化操作）
        new_distances = distance_matrix[next_idx]
        min_distances = torch.min(min_distances, new_distances)
        min_distances[selected] = 0  # 已选点距离保持为0

        if step % log_interval == 0 or step < 10:
            log.info(f"FPS第{step}步: 选择点{next_idx}, 最大最小距离={max_distance:.4f}")

    log.info(f"GPU FPS完成，选择了{len(selected_indices)}个样本")
    return selected_indices


def diversity_based_selection(
    scored_data: List[dict],
    all_logits_by_dataset: dict,
    selection_percentage: float,
    importance_selection_percentage: float = None,
    enable_diversity: bool = True,
    device: torch.device = None,
    distance_batch_size: int = 1000,
    fps_log_interval: int = 100,
) -> List[dict]:
    """
    基于多样性的数据选择（支持两阶段选择策略）

    Args:
        scored_data: 评分后的数据列表
        all_logits_by_dataset: 按数据集分组的logits张量
        selection_percentage: 最终选择比例
        importance_selection_percentage: 第一阶段质量筛选比例（可选）
        enable_diversity: 是否启用多样性选择（False时回退到质量分数选择）
        device: GPU设备
        distance_batch_size: GPU批处理大小
        fps_log_interval: FPS日志输出间隔

    Returns:
        选择后的数据列表
    """
    total_samples = len(scored_data)
    n_select = int(total_samples * selection_percentage)

    # 检查是否需要两阶段选择策略
    use_two_stage = (
        importance_selection_percentage is not None and importance_selection_percentage > selection_percentage
    )

    if not enable_diversity:
        # 回退到原始的质量分数选择
        scored_data.sort(key=lambda x: x["scores"], reverse=True)
        return scored_data[:n_select]

    if use_two_stage:
        # 两阶段选择策略
        log = logging.getLogger(__name__)
        log.info("启用两阶段选择策略:")
        log.info(f"  - 第一阶段: 基于质量分数选择前 {importance_selection_percentage * 100:.1f}% 数据")
        log.info(f"  - 第二阶段: 基于多样性从高质量数据中选择 {selection_percentage * 100:.1f}% 数据")

        # 第一阶段：基于质量分数预筛选
        n_importance = int(total_samples * importance_selection_percentage)
        scored_data.sort(key=lambda x: x["scores"], reverse=True)
        high_quality_data = scored_data[:n_importance]

        log.info(f"  - 第一阶段完成: 从 {total_samples} 个样本中选择了 {len(high_quality_data)} 个高质量样本")
        log.info(f"  - 质量分数范围: {high_quality_data[0]['scores']:.6f} ~ {high_quality_data[-1]['scores']:.6f}")

        # 调整logits数据对应关系
        filtered_logits_by_dataset = {name: [] for name in all_logits_by_dataset.keys()}
        dataset_logits_index = {name: 0 for name in all_logits_by_dataset.keys()}

        # 重新映射高质量数据的logits
        for original_idx, data_item in enumerate(scored_data):
            if original_idx < n_importance:  # 在高质量数据范围内
                dataset_name = data_item["dataset"]
                current_idx = dataset_logits_index[dataset_name]

                if current_idx < len(all_logits_by_dataset[dataset_name]):
                    logits_tensor = all_logits_by_dataset[dataset_name][current_idx]
                    filtered_logits_by_dataset[dataset_name].append(logits_tensor)

                dataset_logits_index[dataset_name] += 1

        # 第二阶段：从高质量数据中进行多样性选择
        log.info(f"  - 第二阶段: 从 {len(high_quality_data)} 个高质量样本中进行多样性选择...")

        # 计算第二阶段的实际选择比例
        stage2_selection_ratio = n_select / len(high_quality_data)
        log.info(f"  - 第二阶段选择比例: {stage2_selection_ratio * 100:.1f}% ({n_select}/{len(high_quality_data)})")

        selected_data = _perform_diversity_selection(
            high_quality_data,
            filtered_logits_by_dataset,
            stage2_selection_ratio,
            device,
            distance_batch_size,
            fps_log_interval,
        )

        log.info(f"两阶段选择完成: 最终选择了 {len(selected_data)} 个样本")
        return selected_data

    else:
        # 单阶段多样性选择（原始逻辑）
        log = logging.getLogger(__name__)
        log.info(f"启用单阶段多样性选择: 从 {total_samples} 个样本中选择 {n_select} 个")

        selected_data = _perform_diversity_selection(
            scored_data, all_logits_by_dataset, selection_percentage, device, distance_batch_size, fps_log_interval
        )

        return selected_data


def _perform_diversity_selection(
    scored_data: List[dict],
    all_logits_by_dataset: dict,
    selection_percentage: float,
    device: torch.device,
    distance_batch_size: int,
    fps_log_interval: int = 100,
) -> List[dict]:
    """
    执行多样性选择的核心逻辑

    Args:
        scored_data: 待选择的数据列表
        all_logits_by_dataset: logits张量数据
        selection_percentage: 选择比例
        device: GPU设备
        distance_batch_size: GPU批处理大小
        fps_log_interval: FPS日志输出间隔

    Returns:
        选择后的数据列表
    """
    total_samples = len(scored_data)
    n_select = int(total_samples * selection_percentage)

    # 1. 收集所有logits张量 - 维度: [样本数, L, E]
    all_logits = []
    sample_to_data_mapping = []

    # 按scored_data的顺序重新组织logits
    dataset_logits_index = {name: 0 for name in all_logits_by_dataset.keys()}

    for data_item in scored_data:
        dataset_name = data_item["dataset"]
        current_idx = dataset_logits_index[dataset_name]

        if current_idx < len(all_logits_by_dataset[dataset_name]):
            logits_tensor = all_logits_by_dataset[dataset_name][current_idx]
            all_logits.append(logits_tensor)
            sample_to_data_mapping.append(data_item)
            dataset_logits_index[dataset_name] += 1
        else:
            log = logging.getLogger(__name__)
            log.warning(f"数据集 {dataset_name} 的logits不足")

    if len(all_logits) != len(scored_data):
        log = logging.getLogger(__name__)
        log.warning(f"logits数量({len(all_logits)}) 与数据数量({len(scored_data)})不匹配")
        # 回退到质量分数选择
        scored_data.sort(key=lambda x: x["scores"], reverse=True)
        return scored_data[:n_select]

    log = logging.getLogger(__name__)
    log.info(f"收集到 {len(all_logits)} 个logits张量")
    log.info(f"张量形状示例: {all_logits[0].shape} (应为 [L, E])")

    # 2. 计算Wasserstein距离矩阵
    distance_matrix = compute_wasserstein_distance_matrix(
        all_logits, device=device, batch_size=distance_batch_size
    )

    # 3. 使用GPU FPS选择多样化样本，传递质量分数用于初始点选择
    sample_quality_scores = [item["scores"] for item in sample_to_data_mapping]
    selected_indices = farthest_point_sampling_gpu(
        distance_matrix, n_select, quality_scores=sample_quality_scores, log_interval=fps_log_interval
    )

    # 4. 根据选择的索引返回对应的数据
    selected_data = [sample_to_data_mapping[idx] for idx in selected_indices]

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
    log.info(f"  - 质量门logits形状: {router_data['quality_logits'].shape}")
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
        "quality_logits": router_data["quality_logits"][position],  # [num_layers, 2]
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
    # 自动选择设备，正确响应CUDA_VISIBLE_DEVICES
    if torch.cuda.is_available():
        # 检查实际可用的GPU设备数量
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            # 使用第一个可用的GPU（在CUDA_VISIBLE_DEVICES环境下会是指定的设备）
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
    # 进行一次前向传播验证模型结构
    with torch.no_grad():
        # 获取模型实际所在的设备
        model_device = next(model.parameters()).device
        dummy_input = torch.ones(1, 10, dtype=torch.long, device=model_device)
        dummy_outputs = model(dummy_input, output_router_logits=True)

        # 验证输出格式
        if not isinstance(dummy_outputs.router_logits[0], dict):
            raise ValueError("模型不是新的两层路由架构，请使用正确的模型")

        if "quality_logits" not in dummy_outputs.router_logits[0]:
            raise ValueError("模型缺少质量门输出，请使用正确的两层路由架构")

        if "moe_logits" not in dummy_outputs.router_logits[0]:
            raise ValueError("模型缺少MoE路由输出，请使用正确的两层路由架构")

    log.info("✓ 验证模型为新的两层路由架构")
    log.info(f"✓ 模型层数: {num_hidden_layers}")
    log.info(f"✓ 质量门输出维度: {dummy_outputs.router_logits[0]['quality_logits'].shape[-1]}")
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
    log.info("✓ 数据集已准备完毕，每个样本包含唯一ID用于索引对应")

    # 4. 数据评分和路由logits记录
    scored_data = []
    # 重构：同时记录质量门和MoE路由的输出
    all_router_data_by_dataset = {
        name: {
            "quality_logits": [],  # 质量门logits: [样本数, 层数, 序列长度, 2]
            "moe_logits": [],  # MoE路由logits: [样本数, 层数, 专家数] (平均后)
            "sample_ids": [],  # 样本的原始ID（唯一标识符）
        }
        for name in cfg.dataset.dataset_names
    }
    dataset_sample_counts = {name: 0 for name in cfg.dataset.dataset_names}  # 记录每个数据集的样本数

    # 创建数据加载器
    def collate_fn(batch):
        """自定义collate函数，保留数据集信息"""
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
                # 格式化为对话文本
                text_parts = []
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    text_parts.append(f"{role}: {content}")
                texts.append("\n".join(text_parts))
            else:
                texts.append("")  # 空文本作为fallback

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
        # 将输入数据移动到模型所在的设备
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_router_logits=True)

        # 处理router logits - 新架构返回字典格式
        batch_size = len(texts)
        sequence_length = inputs["input_ids"].shape[1]

        # 从字典格式中提取MoE路由logits
        moe_router_logits = [layer_dict["moe_logits"] for layer_dict in outputs.router_logits]
        all_router_logits = torch.stack(moe_router_logits)

        if i == 0:
            log.info(f"Router logits形状: {all_router_logits.shape}")
            log.info(f"批次大小: {batch_size}, 序列长度: {sequence_length}")

        # 计算质量分数 - 使用新架构的质量门输出
        for j in range(batch_size):
            dataset_name = dataset_names[j]
            sample_attention_mask = inputs["attention_mask"][j]
            debug_mode = i == 0 and j < 3  # 只对前3个样本启用调试

            # 使用质量门输出计算质量分数
            quality_logits_list = [layer_dict["quality_logits"][j] for layer_dict in outputs.router_logits]
            quality_score = calculate_quality_score_from_gates(
                quality_logits_list,
                sample_attention_mask,
                debug=debug_mode,
            )

            # 获取有效token的掩码信息
            valid_mask = sample_attention_mask.bool()
            actual_length = valid_mask.sum().item()

            # 记录完整的路由信息用于分析和多样性采样
            # 1. 收集质量门logits并在序列维度进行平均 - 与MoE路由保持一致的形状
            quality_logits_averaged = []
            for layer_quality_logits in quality_logits_list:
                # layer_quality_logits: [seq_len, 2]
                if actual_length > 0:
                    # 使用attention_mask提取有效token的logits
                    valid_quality_logits = layer_quality_logits[valid_mask]  # [actual_length, 2]
                    # 对有效token求平均
                    layer_avg_quality_logits = valid_quality_logits.mean(dim=0)  # [2]
                else:
                    # 如果没有有效token，使用零值
                    layer_avg_quality_logits = torch.zeros(2)

                quality_logits_averaged.append(layer_avg_quality_logits)

            # 堆叠为 [num_layers, 2]
            quality_logits_sample = torch.stack(quality_logits_averaged)  # [num_layers, 2]

            # 2. 收集MoE路由logits并计算平均概率分布
            if actual_length > 0:
                moe_logits_list = []
                seq_len = inputs["input_ids"].shape[1]  # 序列长度

                for _, layer_dict in enumerate(outputs.router_logits):
                    # layer_dict["moe_logits"] 形状: [batch*seq_len, num_experts]
                    layer_moe_logits = layer_dict["moe_logits"]  # [batch*seq_len, num_experts]

                    # 提取第j个样本的logits
                    start_idx = j * seq_len
                    end_idx = (j + 1) * seq_len
                    sample_layer_logits = layer_moe_logits[start_idx:end_idx]  # [seq_len, num_experts]

                    # 转换为概率分布
                    sample_layer_probs = torch.softmax(sample_layer_logits, dim=-1)  # [seq_len, num_experts]

                    # 根据attention mask提取有效token的概率，然后求平均
                    valid_probs = sample_layer_probs[valid_mask]  # [actual_length, num_experts]
                    if valid_probs.shape[0] > 0:
                        layer_avg_probs = valid_probs.mean(dim=0)  # [num_experts]
                    else:
                        # 如果没有有效token，使用均匀分布
                        num_experts = sample_layer_probs.shape[-1]
                        layer_avg_probs = torch.ones(num_experts) / num_experts

                    moe_logits_list.append(layer_avg_probs)

                # 构建样本的MoE路由特征矩阵 [num_layers, num_experts]
                sample_moe_matrix = torch.stack(moe_logits_list)  # [num_layers, num_experts]
            else:
                # 如果没有有效token，使用均匀分布
                num_layers = len(outputs.router_logits)
                num_experts = outputs.router_logits[0]["moe_logits"].shape[-1]
                sample_moe_matrix = torch.ones(num_layers, num_experts) / num_experts
                quality_logits_sample = torch.zeros(num_layers, 2)  # 修正：[num_layers, 2]

            # 3. 保存到数据结构中，建立ID对应关系
            all_router_data_by_dataset[dataset_name]["quality_logits"].append(quality_logits_sample.cpu())
            all_router_data_by_dataset[dataset_name]["moe_logits"].append(sample_moe_matrix.cpu())
            all_router_data_by_dataset[dataset_name]["sample_ids"].append(ids[j])

            dataset_sample_counts[dataset_name] += 1

            if i == 0 and j < 3:
                current_sample_index = i * cfg.data_process.batch_size + j
                log.info(f"--- 样本 {current_sample_index} 详细计算过程 ---")
                log.info(f"  - 数据集: {dataset_names[j]}")
                log.info(f"  - 样本ID: {ids[j]}")
                log.info(f"  - 实际长度: {actual_length} / 总长度: {sample_attention_mask.shape[0]}")
                log.info(f"  - 质量门输出形状: {quality_logits_list[0].shape}")
                log.info(f"  - 质量门张量形状: {quality_logits_sample.shape}")
                log.info(f"  - MoE路由矩阵形状: {sample_moe_matrix.shape}")

                # 显示质量门和MoE路由的详细信息
                log.info(f"  - Quality Gate前3层的输出形状: {[ql.shape for ql in quality_logits_list[:3]]}")
                log.info(
                    f"  - MoE Logits前3层的输出形状: {[outputs.router_logits[i]['moe_logits'][j * sequence_length : (j + 1) * sequence_length].shape for i in range(min(3, len(outputs.router_logits)))]}"
                )

                log.info(f"  - 最终质量分数: {quality_score:.6f}")
                log.info("=" * 80)

            scored_data.append(
                {
                    "dataset": dataset_names[j],
                    "id": ids[j],
                    "scores": quality_score,
                    "messages": messages_list[j],
                }
            )

    # 5. 立即保存完整的路由张量文件（包含质量门和MoE路由输出）
    log.info("模型推理完成，立即保存完整路由张量文件...")
    output_path = hydra.utils.to_absolute_path(cfg.output_path)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    router_data_dir = os.path.join(output_dir, "router_data")
    os.makedirs(router_data_dir, exist_ok=True)

    for dataset_name in cfg.dataset.dataset_names:
        dataset_router_data = all_router_data_by_dataset[dataset_name]

        if dataset_router_data["quality_logits"] and dataset_router_data["moe_logits"]:
            # 构建完整的数据字典 - 两种路由数据形状现在保持一致
            router_data_dict = {
                "quality_logits": torch.stack(dataset_router_data["quality_logits"]),  # [N, L, 2]
                "moe_logits": torch.stack(dataset_router_data["moe_logits"]),  # [N, L, E]
                "sample_ids": dataset_router_data["sample_ids"],
                "dataset_name": dataset_name,
                "num_samples": dataset_sample_counts[dataset_name],
                "metadata": {
                    "description": "完整的路由数据，包含质量门和MoE路由输出",
                    "quality_logits_shape": "[N, num_layers, 2] - 质量门平均概率",
                    "moe_logits_shape": "[N, num_layers, num_experts] - MoE路由平均概率",
                    "sample_ids": "样本的唯一ID标识，可与原数据集精确对应",
                    "id_format": 'dataset_name + "_" + number (例如: oasst1_25460)',
                    "note": "两种路由数据均已在序列维度进行平均，形状保持一致",
                },
            }

            # 保存完整数据字典
            router_data_path = os.path.join(router_data_dir, f"{dataset_name}_router_data.pt")
            torch.save(router_data_dict, router_data_path)

            log.info(f"数据集 '{dataset_name}' 的完整路由数据已保存到: {router_data_path}")
            log.info(f"  - 质量门logits形状: {router_data_dict['quality_logits'].shape}")
            log.info(f"  - MoE路由logits形状: {router_data_dict['moe_logits'].shape}")
            log.info(f"  - 样本数: {router_data_dict['num_samples']}")
            sample_ids_preview = dataset_router_data['sample_ids'][:3] if dataset_router_data['sample_ids'] else '无'
            log.info(f"  - 样本ID示例: {sample_ids_preview}")

        else:
            log.warning(f"数据集 '{dataset_name}' 没有路由数据")

    # 释放模型和GPU内存，为距离计算腾出空间
    log.info("释放模型实例和GPU内存...")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log.info("模型内存已释放")

    # 6. 尝试进行多样性数据选择
    log.info("开始多样性数据选择...")

    # 检查是否启用多样性选择
    enable_diversity = getattr(cfg, "enable_diversity_selection", True)
    log.info(f"多样性选择模式: {'启用' if enable_diversity else '禁用(使用质量分数)'}")

    # 使用多样性选择算法
    distance_batch_size = getattr(cfg.distance_computation, "distance_batch_size", 1000)
    fps_log_interval = getattr(cfg.distance_computation, "fps_log_interval", 100)
    importance_selection_percentage = getattr(cfg, "importance_selection_percentage", None)

    # 直接使用路由数据进行多样性选择
    all_logits_by_dataset = {name: all_router_data_by_dataset[name]["moe_logits"] for name in cfg.dataset.dataset_names}

    try:
        selected_data = diversity_based_selection(
            scored_data=scored_data,
            all_logits_by_dataset=all_logits_by_dataset,
            selection_percentage=cfg.selection_percentage,
            importance_selection_percentage=importance_selection_percentage,
            enable_diversity=enable_diversity,
            device=device,
            distance_batch_size=distance_batch_size,
            fps_log_interval=fps_log_interval,
        )
    except torch.OutOfMemoryError as e:
        log.error(f"GPU内存不足，距离计算失败: {e}")
        log.error("但是，宝贵的logits张量已经安全保存！")
        log.info("您可以使用独立脚本继续数据选择过程:")
        log.info(f"  python scripts/continue_selection.py --router_data_dir {router_data_dir} "
                 f"--output_path {output_path} --selection_percentage {cfg.selection_percentage}")
        if importance_selection_percentage:
            log.info(f"  添加参数: --importance_selection_percentage "
                     f"{importance_selection_percentage}")
        if not enable_diversity:
            log.info("  添加参数: --disable_diversity")
        log.info(f"  添加参数: --distance_batch_size {distance_batch_size}")
        raise
    except Exception as e:
        log.error(f"数据选择过程中发生错误: {e}")
        log.error("但是，宝贵的logits张量已经安全保存！")
        log.info("您可以使用独立脚本继续数据选择过程:")
        log.info(f"  python scripts/continue_selection.py --router_data_dir {router_data_dir} "
                 f"--output_path {output_path}")
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
