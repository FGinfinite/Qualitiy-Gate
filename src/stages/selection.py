# src/stages/selection.py
import json
import logging
import os
from typing import Tuple

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data import load_local_datasets
from src.models.select_moe import SelectMoeForCausalLM, register_select_moe
from src.training.full_rank_finetuning import load_full_rank_weights


def get_model_and_tokenizer(
    cfg: DictConfig, device: torch.device
) -> Tuple[SelectMoeForCausalLM, AutoTokenizer]:
    """
    加载预训练的Select-MoE模型并应用全秩微调权重
    """
    # 注册Select-MoE模型
    register_select_moe()

    model_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }

    # 加载预转换的Select-MoE模型
    model = SelectMoeForCausalLM.from_pretrained(
        cfg.selector_model.path, **model_kwargs
    )

    # 加载全秩微调权重
    load_full_rank_weights(model, cfg.model_checkpoint_path)

    model.to(device)
    model.eval()

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(cfg.selector_model.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def calculate_quality_score(
    layer_token_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    original_num_experts: int,
    debug: bool = False,
) -> float:
    """
    计算单个数据点的质量分数，基于激活好专家的平均比例

    Args:
        layer_token_logits: 形状为 (num_layers, seq_len, num_experts) 的logits张量
        attention_mask: 形状为 (seq_len,) 的注意力掩码，1表示实际token，0表示填充token
        original_num_experts: 原始（好）专家的数量
        debug: 是否打印调试信息

    Returns:
        质量分数：好专家概率的平均值
    """
    # 获取实际token的掩码
    valid_mask = attention_mask.bool()  # (seq_len,)
    actual_length = valid_mask.sum().item()

    if debug:
        print(f"    [函数内部] 输入logits形状: {layer_token_logits.shape}")
        print(f"    [函数内部] attention_mask形状: {attention_mask.shape}")
        print(f"    [函数内部] 实际token数量: {actual_length}")
        print(f"    [函数内部] 原始专家数量: {original_num_experts}")

    if actual_length == 0:
        if debug:
            print("    [函数内部] 没有实际token，返回0.0")
        return 0.0

    # 只处理实际token位置的logits
    valid_logits = layer_token_logits[
        :, valid_mask, :
    ]  # (num_layers, actual_length, num_experts)

    if debug:
        print(f"    [函数内部] 提取有效logits后形状: {valid_logits.shape}")
        print(
            f"    [函数内部] 第1层第1个token的logits范围: [{valid_logits[0, 0, :].min().item():.4f}, {valid_logits[0, 0, :].max().item():.4f}]"
        )

    # 对每一层每个token的logits应用softmax，得到概率分布
    token_probs = torch.softmax(
        valid_logits, dim=-1
    )  # (num_layers, actual_length, num_experts)

    if debug:
        print(f"    [函数内部] Softmax后概率形状: {token_probs.shape}")
        print(
            f"    [函数内部] 第1层第1个token的概率范围: [{token_probs[0, 0, :].min().item():.6f}, {token_probs[0, 0, :].max().item():.6f}]"
        )
        print(
            f"    [函数内部] 第1层第1个token的概率总和: {token_probs[0, 0, :].sum().item():.6f}"
        )

    # 计算每一层每个token中好专家的概率总和
    quality_probs_per_token = token_probs[:, :, :original_num_experts].sum(
        dim=-1
    )  # (num_layers, actual_length)

    if debug:
        print(
            f"    [函数内部] 好专家概率per token形状: {quality_probs_per_token.shape}"
        )
        print(
            f"    [函数内部] 第1层前3个token的好专家概率: {quality_probs_per_token[0, : min(3, actual_length)].tolist()}"
        )

    # 先在层内对所有实际token求平均，得到每层的好专家概率
    quality_probs_per_layer = quality_probs_per_token.mean(dim=1)  # (num_layers,)

    if debug:
        print(f"    [函数内部] 每层平均好专家概率形状: {quality_probs_per_layer.shape}")
        print(f"    [函数内部] 每层平均好专家概率: {quality_probs_per_layer.tolist()}")

    # 再对所有层求平均，得到最终质量分数
    avg_quality_prob = quality_probs_per_layer.mean()

    if debug:
        print(f"    [函数内部] 最终平均质量分数: {avg_quality_prob.item():.6f}")

    return avg_quality_prob.item()


def select(cfg: DictConfig) -> None:
    """
    数据选择阶段的主函数
    """
    log = logging.getLogger(__name__)
    log.info("--- 开始阶段2：数据选择 ---")

    # 确定目标设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"使用设备: {device}")

    # 如果使用GPU，清理缓存
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 1. 加载模型和分词器
    log.info(f"从检查点加载模型: {cfg.model_checkpoint_path}")
    model, tokenizer = get_model_and_tokenizer(cfg, device)
    num_hidden_layers = model.config.num_hidden_layers

    # 2. 定义专家角色
    # 检查模型配置以确定专家数量
    total_experts = model.config.num_experts
    num_experts_per_tok = model.config.num_experts_per_tok  # top-k值

    # 对于Select-MoE模型，我们需要确定原始专家数量
    # 通过检查模型的实际结构来确定
    sample_layer = model.model.layers[0].mlp
    if hasattr(sample_layer, "original_num_experts"):
        original_num_experts = sample_layer.original_num_experts
    else:
        # 如果没有这个属性，使用默认值
        original_num_experts = 64  # OLMoE-1B-7B-0125的原始专家数

    # 实际的总专家数可能与配置不同，我们需要从实际的router logits中获取
    # 先进行一次前向传播来获取实际的专家数量
    with torch.no_grad():
        dummy_input = torch.ones(1, 10, dtype=torch.long, device=device)
        dummy_outputs = model(dummy_input, output_router_logits=True)
        actual_total_experts = dummy_outputs.router_logits[0].shape[-1]

    log.info(f"配置中的总专家数: {total_experts}")
    log.info(f"实际的总专家数: {actual_total_experts}")
    log.info(
        f"原始（好）专家数: {original_num_experts} (索引 0 到 {original_num_experts - 1})"
    )
    log.info(
        f"垃圾桶专家数: {actual_total_experts - original_num_experts} (索引 {original_num_experts} 到 {actual_total_experts - 1})"
    )
    log.info(f"每个token选择的专家数: {num_experts_per_tok}")

    # 使用实际的专家数量
    total_experts = actual_total_experts

    # 3. 加载和准备数据集
    log.info(f"加载数据集: {cfg.dataset.dataset_names}")
    dataset = load_local_datasets(
        data_dir=cfg.dataset.data_dir,
        dataset_names=cfg.dataset.dataset_names,
        sample_percentage=cfg.dataset.subset_ratio,
        seed=cfg.dataset.seed,
    )

    if cfg.dataset.shuffle:
        dataset = dataset.shuffle(seed=cfg.dataset.seed)

    log.info(f"总样本数: {len(dataset)}")

    # 4. 数据评分和logits记录
    scored_data = []
    all_logits_by_dataset = {
        name: [] for name in cfg.dataset.dataset_names
    }  # 按数据集记录logits
    dataset_sample_counts = {
        name: 0 for name in cfg.dataset.dataset_names
    }  # 记录每个数据集的样本数

    # 创建数据加载器
    def collate_fn(batch):
        """自定义collate函数，保留数据集信息"""
        return {
            "messages": [item["messages"] for item in batch],
            "dataset": [item["dataset"] for item in batch],
            "id": [item["id"] for item in batch],
        }

    dataloader = DataLoader(
        dataset, batch_size=cfg.data_process.batch_size, collate_fn=collate_fn
    )

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
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_router_logits=True)

        # 处理router logits
        batch_size = len(texts)
        sequence_length = inputs["input_ids"].shape[1]

        # 堆叠所有层的router logits: (num_layers, bs * seq_len, num_experts)
        all_router_logits = torch.stack(outputs.router_logits)

        if i == 0:
            log.info(f"Router logits形状: {all_router_logits.shape}")
            log.info(
                f"期望形状: ({num_hidden_layers}, {batch_size * sequence_length}, {total_experts})"
            )
            log.info(f"批次大小: {batch_size}, 序列长度: {sequence_length}")

        # 重塑为: (num_layers, bs, seq_len, num_experts)
        reshaped_logits = all_router_logits.view(
            num_hidden_layers, batch_size, sequence_length, total_experts
        )

        # 计算质量分数（使用每层每个token的softmax概率，排除填充token）
        for j in range(batch_size):
            dataset_name = dataset_names[j]

            # 获取当前样本的每层每个token的logits: (num_layers, seq_len, num_experts)
            sample_layer_token_logits = reshaped_logits[:, j, :, :]

            # 获取当前样本的注意力掩码
            sample_attention_mask = inputs["attention_mask"][j]

            # 计算质量分数（考虑填充token）
            debug_mode = i == 0 and j < 3  # 只对前3个样本启用调试
            quality_score = calculate_quality_score(
                sample_layer_token_logits,
                sample_attention_mask,
                original_num_experts,
                debug=debug_mode,
            )

            # 记录原始logits（只对实际token位置求平均）
            valid_mask = sample_attention_mask.bool()
            actual_length = valid_mask.sum().item()
            if actual_length > 0:
                # 只对实际token位置的logits求平均
                sample_logits = sample_layer_token_logits[:, valid_mask, :].mean(
                    dim=1
                )  # (num_layers, num_experts)
            else:
                # 如果没有实际token，使用全零
                sample_logits = torch.zeros(
                    sample_layer_token_logits.shape[0],
                    sample_layer_token_logits.shape[2],
                )

            all_logits_by_dataset[dataset_name].append(sample_logits.cpu())
            dataset_sample_counts[dataset_name] += 1

            if i == 0 and j < 3:
                log.info(
                    f"--- 样本 {i * cfg.data_process.batch_size + j} 详细计算过程 ---"
                )
                log.info(f"  - 数据集: {dataset_names[j]}")
                log.info(f"  - ID: {ids[j]}")
                log.info(
                    f"  - 实际长度: {actual_length} / 总长度: {sample_attention_mask.shape[0]}"
                )
                log.info(
                    f"  - 原始层级Token Logits形状: {sample_layer_token_logits.shape}"
                )

                # 显示attention mask的详细信息
                log.info(
                    f"  - Attention Mask前10个值: {sample_attention_mask[:10].tolist()}"
                )
                log.info(
                    f"  - Attention Mask后10个值: {sample_attention_mask[-10:].tolist()}"
                )

                if actual_length > 0:
                    # 步骤1: 提取有效token的logits
                    valid_logits = sample_layer_token_logits[:, valid_mask, :]
                    log.info(
                        f"  - 步骤1: 提取有效token后的logits形状: {valid_logits.shape}"
                    )
                    log.info(
                        f"    (num_layers={valid_logits.shape[0]}, actual_tokens={valid_logits.shape[1]}, num_experts={valid_logits.shape[2]})"
                    )

                    # 显示第一层前5个token的原始logits示例
                    first_layer_first_tokens = valid_logits[
                        0, : min(5, actual_length), :
                    ]
                    log.info(
                        f"  - 第1层前{min(5, actual_length)}个token的原始logits形状: {first_layer_first_tokens.shape}"
                    )
                    log.info(
                        f"  - 第1层第1个token的前10个专家logits: {first_layer_first_tokens[0, :10].tolist()}"
                    )
                    log.info(
                        f"  - 第1层第1个token的后10个专家logits: {first_layer_first_tokens[0, -10:].tolist()}"
                    )

                    # 步骤2: 应用softmax得到概率
                    token_probs = torch.softmax(valid_logits, dim=-1)
                    log.info(f"  - 步骤2: Softmax后的概率形状: {token_probs.shape}")

                    # 显示第一层第一个token的概率分布
                    first_token_probs = token_probs[0, 0, :]
                    log.info("  - 第1层第1个token的概率分布:")
                    log.info(f"    - 前10个专家概率: {first_token_probs[:10].tolist()}")
                    log.info(
                        f"    - 后10个专家概率: {first_token_probs[-10:].tolist()}"
                    )
                    log.info(f"    - 概率总和: {first_token_probs.sum().item():.6f}")

                    # 步骤3: 计算好专家概率
                    quality_probs_per_token = token_probs[
                        :, :, :original_num_experts
                    ].sum(dim=-1)
                    log.info(
                        f"  - 步骤3: 每层每个token的好专家概率形状: {quality_probs_per_token.shape}"
                    )
                    log.info(
                        f"    (num_layers={quality_probs_per_token.shape[0]}, actual_tokens={quality_probs_per_token.shape[1]})"
                    )

                    # 显示前几层前几个token的好专家概率
                    log.info("  - 前3层前5个token的好专家概率:")
                    for layer_idx in range(min(3, quality_probs_per_token.shape[0])):
                        layer_token_probs = quality_probs_per_token[
                            layer_idx, : min(5, actual_length)
                        ]
                        log.info(
                            f"    - 第{layer_idx + 1}层: {layer_token_probs.tolist()}"
                        )

                    # 步骤4: 层内平均（对所有实际token）
                    quality_probs_per_layer = quality_probs_per_token.mean(dim=1)
                    log.info(
                        f"  - 步骤4: 每层的平均好专家概率形状: {quality_probs_per_layer.shape}"
                    )
                    log.info(
                        f"  - 每层的平均好专家概率: {quality_probs_per_layer.tolist()}"
                    )

                    # 步骤5: 层间平均
                    final_score = quality_probs_per_layer.mean().item()
                    log.info(
                        f"  - 步骤5: 最终质量分数 = {quality_probs_per_layer.sum().item():.6f} / {len(quality_probs_per_layer)} = {final_score:.6f}"
                    )

                    # 验证计算
                    log.info(f"  - 验证: 函数返回的质量分数: {quality_score:.6f}")
                    log.info(
                        f"  - 验证: 计算是否一致: {'✅' if abs(final_score - quality_score) < 1e-6 else '❌'}"
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

    # 5. 筛选和保存数据
    log.info("筛选和保存数据...")
    scored_data.sort(key=lambda x: x["scores"], reverse=True)
    num_to_select = int(len(scored_data) * cfg.selection_percentage)
    selected_data = scored_data[:num_to_select]

    log.info(
        f"选择了前 {len(selected_data)} 个样本 ({cfg.selection_percentage * 100:.2f}%)"
    )
    if len(selected_data) > 3:
        log.info(f"前3个分数: {[d['scores'] for d in selected_data[:3]]}")
        log.info(f"后3个分数: {[d['scores'] for d in selected_data[-3:]]}")

    # 保存选择的数据
    output_path = hydra.utils.to_absolute_path(cfg.output_path)
    log.info(f"确保输出目录存在: {output_path}")

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in selected_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    log.info(f"筛选后的数据已保存到: {output_path}")

    # 6. 保存logits张量文件
    log.info("保存logits张量文件...")
    logits_dir = os.path.join(output_dir, "logits")
    os.makedirs(logits_dir, exist_ok=True)

    for dataset_name in cfg.dataset.dataset_names:
        if (
            dataset_name in all_logits_by_dataset
            and all_logits_by_dataset[dataset_name]
        ):
            # 将列表中的张量堆叠成一个大张量
            dataset_logits = torch.stack(all_logits_by_dataset[dataset_name])
            logits_path = os.path.join(logits_dir, f"{dataset_name}_logits.pt")
            torch.save(dataset_logits, logits_path)
            log.info(f"数据集 '{dataset_name}' 的logits已保存到: {logits_path}")
            log.info(f"  - 形状: {dataset_logits.shape}")
            log.info(f"  - 样本数: {dataset_sample_counts[dataset_name]}")
        else:
            log.warning(f"数据集 '{dataset_name}' 没有logits数据")

    log.info("--- 阶段2：数据选择完成 ---")
