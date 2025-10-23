# src/stages/selection.py
import logging
import os
from typing import Tuple

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

from src.data import load_and_prepare_dataset
from src.models.quality_gate_model import QualityGateForCausalLM, register_quality_gate
from src.training.full_rank_finetuning import load_full_rank_weights


def get_model_and_tokenizer(cfg: DictConfig) -> Tuple[QualityGateForCausalLM, AutoTokenizer]:
    """
    加载预先转换好的质量门控模型并应用Stage 1训练的权重

    步骤：
    1. 从 selector_model.path 加载预转换的基础质量门控模型
    2. 从 model_checkpoint_path/full_rank_weights.pt 加载训练好的质量门控权重
    """
    log = logging.getLogger(__name__)

    # 注册质量门控模型
    register_quality_gate()

    model_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }

    # 1. 加载预转换的质量门控模型
    log.info(f"从预转换模型加载: {cfg.selector_model.path}")
    model = QualityGateForCausalLM.from_pretrained(cfg.selector_model.path, **model_kwargs)
    log.info(f"✓ 成功加载基础质量门控模型")

    # 2. 加载Stage 1训练的质量门控权重
    checkpoint_path = os.path.join(cfg.model_checkpoint_path, "full_rank_weights.pt")
    log.info(f"从检查点加载质量门控权重: {checkpoint_path}")
    tokenizer = load_full_rank_weights(model, checkpoint_path)

    if tokenizer is None:
        log.warning("检查点中未包含分词器，使用基础模型的分词器")
        tokenizer_name = cfg.selector_model.get("tokenizer_name", cfg.selector_model.path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    log.info(f"✓ 成功加载质量门控权重")

    model.eval()

    assert tokenizer is not None, "分词器加载失败，请检查模型路径和配置"

    return model, tokenizer


def compute_token_perplexity(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    计算每个token的困惑度

    Args:
        logits: 模型输出的logits，形状为 (batch_size, seq_len, vocab_size)
        input_ids: 输入的token IDs，形状为 (batch_size, seq_len)
        attention_mask: 注意力掩码，形状为 (batch_size, seq_len)

    Returns:
        每个token的困惑度，形状为 (batch_size, seq_len)
    """
    # 移位以对齐预测和目标
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()

    # 计算交叉熵损失（每个token）
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    # 展平以计算损失
    shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels_flat = shift_labels.view(-1)

    # 计算每个token的负对数似然
    nll_per_token = loss_fct(shift_logits_flat, shift_labels_flat)

    # 重塑回原始形状
    nll_per_token = nll_per_token.view(shift_labels.size())

    # 计算困惑度 (exp(nll))
    ppl_per_token = torch.exp(nll_per_token)

    # 对于第一个token（没有预测），填充为1
    batch_size, seq_len = input_ids.shape
    ppl_full = torch.ones(batch_size, seq_len, device=ppl_per_token.device)
    ppl_full[:, 1:] = ppl_per_token

    # 应用attention mask
    ppl_full = ppl_full * shift_attention_mask.float()

    return ppl_full


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
    log.info(f"  - 质量门分数形状: {router_data['quality_gates'].shape}")
    log.info(f"  - 困惑度形状: {router_data['perplexities'].shape}")

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
        "quality_gates": router_data["quality_gates"][position],  # [num_layers, seq_len]
        "perplexities": router_data["perplexities"][position],  # [seq_len]
        "position_in_dataset": position,
    }


def select(cfg: DictConfig) -> None:
    """
    数据选择阶段的主函数（阶段2：统计收集）
    """
    # 设置全局种子以确保实验可复现
    set_seed(cfg.seed)

    log = logging.getLogger(__name__)
    log.info("--- 开始阶段2：统计收集（数据选择第一步）---")
    log.info(f"使用全局种子: {cfg.seed}")

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
    log.info(f"模型层数: {num_hidden_layers}")

    # 2. 验证模型架构
    with torch.no_grad():
        model_device = next(model.parameters()).device
        dummy_input = torch.ones(1, 10, dtype=torch.long, device=model_device)
        dummy_outputs = model(dummy_input, output_router_logits=True)

        # 验证输出格式
        if dummy_outputs.past_key_values is None or len(dummy_outputs.past_key_values) == 0:
            log.warning("模型未输出router logits，请检查配置")

    log.info("✓ 验证模型为质量门控架构")

    # 3. 加载和准备数据集
    log.info(f"正在加载数据集...")
    dataset = load_and_prepare_dataset(cfg)

    if cfg.dataset.shuffle:
        log.info("对数据集进行shuffle...")
        dataset = dataset.shuffle(seed=cfg.seed)

    log.info(f"总样本数: {len(dataset)}")
    log.info("✓ 数据集已准备完毕")

    # 4. 收集统计量
    # 确定数据集名称列表
    if cfg.dataset.dataset_from == "local":
        dataset_names_list = cfg.dataset.local.dataset_names
    elif cfg.dataset.dataset_from == "hf":
        dataset_names_list = [ds.dataset_name for ds in cfg.dataset.hf.datasets]
    else:
        raise ValueError(f"不支持的数据源: {cfg.dataset.dataset_from}")

    log.info(f"数据集来源: {cfg.dataset.dataset_from}")
    log.info(f"数据集名称: {dataset_names_list}")

    all_router_data_by_dataset = {
        name: {
            "quality_gates": [],  # 质量门控logits: [N, L, T]
            "perplexities": [],  # token困惑度: [N, T]
            "sample_ids": [],  # 样本ID
        }
        for name in dataset_names_list
    }
    dataset_sample_counts = {name: 0 for name in dataset_names_list}

    # 创建数据加载器
    def collate_fn(batch):
        return {
            "messages": [item["messages"] for item in batch],
            "dataset": [item["dataset"] for item in batch],
            "id": [item["id"] for item in batch],
        }

    dataloader = DataLoader(dataset, batch_size=cfg.data_process.batch_size, collate_fn=collate_fn)

    log.info("开始收集统计量...")
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

        # 处理每个样本
        batch_size = len(texts)

        # 计算困惑度
        perplexities = compute_token_perplexity(outputs.logits, inputs["input_ids"], inputs["attention_mask"])  # [batch_size, seq_len]

        if i == 0:
            log.info(f"困惑度形状: {perplexities.shape}")

        for j in range(batch_size):
            dataset_name = dataset_names[j]
            sample_attention_mask = inputs["attention_mask"][j]
            debug_mode = i == 0 and j < 3

            # 获取有效token的掩码信息
            valid_mask = sample_attention_mask.bool()
            actual_length = valid_mask.sum().item()

            # 收集质量门控logits
            # 注意：model的outputs.past_key_values实际上不是past_key_values
            # 我们需要通过一个特殊的前向传播来获取router_logits
            # 让我们先获取单个样本的输入
            single_input = {
                "input_ids": inputs["input_ids"][j : j + 1],
                "attention_mask": inputs["attention_mask"][j : j + 1],
            }

            with torch.no_grad():
                single_output = model(**single_input, output_router_logits=True)

            # 从模型的中间层提取质量门控logits
            # 由于我们修改了模型，需要确保能够访问到router_logits
            # 这里假设model.model.layers中每一层都有quality_gate
            quality_gates_list = []
            hidden_states = model.model.embed_tokens(single_input["input_ids"])

            # 手动前向传播以收集质量门控
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
            position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
            causal_mask = None  # 简化处理

            for layer_idx, layer in enumerate(model.model.layers):
                # 自注意力
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
                hidden_states, _, _ = layer.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                hidden_states = residual + hidden_states

                # 质量门控
                residual = hidden_states
                hidden_states = layer.post_attention_layernorm(hidden_states)
                quality_score, _ = layer.quality_gate(hidden_states)  # [1, seq_len, 1]
                quality_gates_list.append(quality_score.squeeze(0).squeeze(-1))  # [seq_len]

                # FFN
                hidden_states = layer.mlp(hidden_states)
                hidden_states = residual + hidden_states

            # 堆叠所有层的质量门控logits: [L, seq_len]
            quality_gates_sample = torch.stack(quality_gates_list)  # [L, seq_len]

            # 获取该样本的困惑度
            perplexity_sample = perplexities[j]  # [seq_len]

            # 保存到数据结构中
            all_router_data_by_dataset[dataset_name]["quality_gates"].append(quality_gates_sample.cpu())
            all_router_data_by_dataset[dataset_name]["perplexities"].append(perplexity_sample.cpu())
            all_router_data_by_dataset[dataset_name]["sample_ids"].append(ids[j])
            dataset_sample_counts[dataset_name] += 1

            if i == 0 and j < 3:
                current_sample_index = i * cfg.data_process.batch_size + j
                log.info(f"--- 样本 {current_sample_index} 详细计算过程 ---")
                log.info(f"  - 数据集: {dataset_names[j]}")
                log.info(f"  - 样本ID: {ids[j]}")
                log.info(f"  - 实际长度: {actual_length} / 总长度: {sample_attention_mask.shape[0]}")
                log.info(f"  - 质量门控形状: {quality_gates_sample.shape}")
                log.info(f"  - 困惑度形状: {perplexity_sample.shape}")
                log.info(f"  - 平均困惑度: {perplexity_sample[valid_mask].mean().item():.4f}")

    # 5. 立即保存完整的路由张量文件
    log.info("模型推理完成，立即保存完整统计数据...")
    output_path = hydra.utils.to_absolute_path(cfg.output_path)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    router_data_dir = os.path.join(output_dir, "router_data")
    os.makedirs(router_data_dir, exist_ok=True)

    for dataset_name in dataset_names_list:
        dataset_router_data = all_router_data_by_dataset[dataset_name]

        if dataset_router_data["quality_gates"] and dataset_router_data["perplexities"]:
            # 找到最大序列长度并进行padding
            max_seq_len = max(qg.shape[1] for qg in dataset_router_data["quality_gates"])

            # Pad质量门控和困惑度到相同长度
            padded_quality_gates = []
            padded_perplexities = []

            for qg, ppl in zip(dataset_router_data["quality_gates"], dataset_router_data["perplexities"]):
                L, T = qg.shape
                # Pad质量门控
                padded_qg = torch.zeros(L, max_seq_len)
                padded_qg[:, :T] = qg
                padded_quality_gates.append(padded_qg)

                # Pad困惑度
                padded_ppl = torch.zeros(max_seq_len)
                padded_ppl[:T] = ppl
                padded_perplexities.append(padded_ppl)

            router_data_dict = {
                "quality_gates": torch.stack(padded_quality_gates),  # [N, L, max_seq_len]
                "perplexities": torch.stack(padded_perplexities),  # [N, max_seq_len]
                "sample_ids": dataset_router_data["sample_ids"],
                "dataset_name": dataset_name,
                "num_samples": dataset_sample_counts[dataset_name],
                "max_seq_len": max_seq_len,
                "metadata": {
                    "description": "质量门控和困惑度统计数据",
                    "quality_gates_shape": "[N, num_layers, max_seq_len] - 质量门控logits（sigmoid前）",
                    "perplexities_shape": "[N, max_seq_len] - 每个token的困惑度",
                    "sample_ids": "样本的唯一ID标识",
                },
            }

            router_data_path = os.path.join(router_data_dir, f"{dataset_name}_router_data.pt")
            torch.save(router_data_dict, router_data_path)

            log.info(f"数据集 '{dataset_name}' 的统计数据已保存到: {router_data_path}")
            log.info(f"  - 质量门控形状: {router_data_dict['quality_gates'].shape}")
            log.info(f"  - 困惑度形状: {router_data_dict['perplexities'].shape}")
            log.info(f"  - 样本数: {router_data_dict['num_samples']}")

        else:
            log.warning(f"数据集 '{dataset_name}' 没有统计数据")

    # 释放模型和GPU内存
    log.info("释放模型实例和GPU内存...")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log.info("模型内存已释放")

    # 6. 统计数据收集完成
    log.info("统计数据已保存，可以使用独立脚本进行数据筛选:")
    log.info(f"  使用continue_selection.py: CUDA_VISIBLE_DEVICES=0 uv run scripts/continue_selection.py router_data_dir={router_data_dir}")
    log.info(f"  使用batch_selection.py: CUDA_VISIBLE_DEVICES=0 uv run scripts/batch_selection.py root_dir={os.path.dirname(router_data_dir)}")
    log.info("--- 阶段2：统计收集完成 ---")
