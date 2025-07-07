# src/modeling.py
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers.models.olmoe.modeling_olmoe import (
    OlmoeSparseMoeBlock,
)

# ---------------------------------------------------------------------------
# 带有“垃圾桶”专家的自定义 MoE 模块
# ---------------------------------------------------------------------------


class TrashCanMoE(OlmoeSparseMoeBlock):
    """
    一个经过修改的 OlmoeSparseMoeBlock，增加了“垃圾桶”专家。
    这些专家不执行任何计算并返回零，作为路由器的负向激励。
    """

    def __init__(
        self,
        config,
        original_moe: OlmoeSparseMoeBlock,
        training_cfg: DictConfig,
    ):
        """
        初始化 TrashCanMoE 模块。

        Args:
            config: 模型配置。
            original_moe: 需要被替换的原始 OlmoeSparseMoeBlock 实例。
            training_cfg (DictConfig): 包含垃圾桶专家初始化参数的训练配置。
        """
        # 因为我们重写了 __init__，所以需要首先调用 nn.Module 的 init 方法
        nn.Module.__init__(self)

        # 从父类的 __init__ 方法中手动设置属性
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # 复制原始的、预训练过的专家
        self.experts = original_moe.experts
        self.original_num_experts = config.num_experts

        # 定义要添加的垃圾桶专家的数量
        self.trash_can_experts_count = self.top_k

        # 专家总数是原始专家和垃圾桶专家数量之和
        new_num_experts = self.original_num_experts + self.trash_can_experts_count
        self.num_experts = new_num_experts

        # 创建一个新的门控层，其输出维度已扩展
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)

        # 从原始门控层复制权重，并确保数据类型一致
        self.gate.weight.data[: self.original_num_experts, :] = (
            original_moe.gate.weight.data.to(self.gate.weight.dtype)
        )

        # 初始化新的“垃圾桶”专家的权重
        # 这些权重将指导路由器何时丢弃一个令牌。
        nn.init.normal_(
            self.gate.weight.data[self.original_num_experts :, :],
            mean=training_cfg.training.trash_can_init_mean,
            std=training_cfg.training.trash_can_init_std,
        )
        # 确保新初始化的权重也具有正确的 dtype
        self.gate.to(original_moe.gate.weight.dtype)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TrashCanMoE 模块的前向传播过程。

        将令牌路由到原始专家或“垃圾桶”，在“垃圾桶”中它们实际上被置零。
        这个实现总是返回 router_logits，以便上层模块可以捕获它们。
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        # 从扩展的门控层获取路由 logits
        router_logits = self.gate(hidden_states_reshaped)

        # 获取 top-k 路由权重和选定的专家
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        routing_weights = routing_weights.to(hidden_states.dtype)

        # 将最终的隐藏状态初始化为零。
        # 被路由到垃圾桶的令牌将保持为零。
        final_hidden_states = torch.zeros_like(hidden_states_reshaped)

        # 创建一个 one-hot 掩码，以标识哪些令牌被路由到哪个专家
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # 只处理原始专家
        for expert_idx in range(self.original_num_experts):
            expert_layer = self.experts[expert_idx]
            # 找到哪些令牌被路由到当前专家
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # 为当前专家选择隐藏状态
            current_state = hidden_states_reshaped[None, top_x].reshape(-1, hidden_dim)

            # 计算专家输出并按路由权重进行缩放
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )

            # 将专家的输出添加到最终的隐藏状态中
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )

        # 始终返回 router_logits 以便上层模块可以捕获它们
        return final_hidden_states, router_logits


# ---------------------------------------------------------------------------
# 模型修改的辅助函数
# ---------------------------------------------------------------------------


def replace_moe_layers_with_trashcan(
    model: nn.Module, config: DictConfig, training_cfg: DictConfig
):
    """
    递归地遍历模型，将所有 OlmoeSparseMoeBlock 实例替换为 TrashCanMoE 实例。

    Args:
        model (nn.Module): 要修改的模型。
        config (DictConfig): 模型配置。
        training_cfg (DictConfig): 包含垃圾桶专家初始化参数的训练配置。
    """
    for name, module in model.named_children():
        if isinstance(module, OlmoeSparseMoeBlock):
            # 创建一个新的 TrashCanMoE 实例来替换原始的 MoE 块
            new_moe = TrashCanMoE(config, module, training_cfg)
            setattr(model, name, new_moe)
        elif len(list(module.children())) > 0:
            # 递归到子模块中
            replace_moe_layers_with_trashcan(module, config, training_cfg)
