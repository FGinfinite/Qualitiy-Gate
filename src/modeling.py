# src/modeling.py
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import OlmoeForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
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
# 新的 CausalLM 模型，带有约束损失（基于前向挂钩）
# ---------------------------------------------------------------------------


class TrashCanMoEForCausalLM(OlmoeForCausalLM):
    """
    一个包装了 OlmoeForCausalLM 的模型，增加了基于“垃圾桶”专家激活的约束损失。
    这个版本使用“前向挂钩”来捕获 router_logits，而不是继承。
    """

    def __init__(
        self,
        config,
        constraint_loss_weight: float = 0.1,
        trash_can_loss_beta: float = 5.0,
    ):
        """
        初始化模型。

        Args:
            config: 模型配置。
            constraint_loss_weight (float): 约束损失的权重。
            trash_can_loss_beta (float): Beta 分布的 beta 参数。
        """
        super().__init__(config)
        self.constraint_loss_weight = constraint_loss_weight
        self.trash_can_loss_beta = trash_can_loss_beta
        # Beta 分布的 alpha 参数固定为 1
        self.trash_can_loss_alpha = 1.0
        self._hook_handles = []
        self._router_logits_list = []

    def _get_router_logits_hook(self, module, input, output):
        """一个内部挂钩函数，用于捕获并存储 router_logits。"""
        # output 是一个元组 (final_hidden_states, router_logits)
        self._router_logits_list.append(output[1])

    def activate_hooks(self):
        """清空现有句柄和 logits，并为所有 MoE 层注册新的前向挂钩。"""
        # 首先确保之前的句柄已被移除且列表为空
        self.deactivate_hooks()
        self._router_logits_list = []

        for layer in self.model.layers:
            handle = layer.mlp.register_forward_hook(self._get_router_logits_hook)
            self._hook_handles.append(handle)

    def deactivate_hooks(self):
        """移除所有已注册的挂钩并清空句柄列表。"""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        模型的前向传播，增加了约束损失的计算。
        这个重构后的版本依赖于外部管理的挂钩。
        """
        # 在每次前向传播开始时，清空 router_logits_list
        self._router_logits_list = []

        # 1. 执行标准的父类前向传播
        # 假设 `activate_hooks` 已经在外部被调用
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        # 2. 如果计算了 Causal LM 损失，则计算并添加约束损失
        if (
            outputs.loss is not None
            and self.constraint_loss_weight > 0
            and self._router_logits_list
        ):
            constraint_loss = torch.tensor(0.0, device=outputs.logits.device)

            for router_logits in self._router_logits_list:
                # router_logits: (batch_size * sequence_length, num_experts)

                # 计算流向垃圾桶专家的概率总和
                all_expert_probs = F.softmax(router_logits, dim=1)
                # 假设垃圾桶专家是最后添加的
                trash_can_probs = all_expert_probs[
                    :, -self.config.num_experts_per_tok :
                ].sum(dim=-1)

                # 我们需要根据 attention_mask 来忽略填充令牌
                if attention_mask is not None:
                    # (batch, seq_len) -> (batch * seq_len)
                    active_tokens_mask = attention_mask.view(-1) == 1
                    masked_probs = trash_can_probs[active_tokens_mask]
                    if masked_probs.numel() > 0:
                        layer_ratio = masked_probs.mean()
                    else:
                        layer_ratio = torch.tensor(0.0, device=outputs.logits.device)
                else:
                    layer_ratio = trash_can_probs.mean()

                # 使用 Beta 分布的负对数似然作为损失
                epsilon = 1e-6
                layer_ratio = torch.clamp(layer_ratio, epsilon, 1.0 - epsilon)

                # L = -log(Beta(ratio | alpha, beta))
                # 我们只关心与 ratio 相关的部分
                layer_constraint_loss = -(
                    (self.trash_can_loss_alpha - 1) * torch.log(layer_ratio)
                    + (self.trash_can_loss_beta - 1) * torch.log(1 - layer_ratio)
                )
                constraint_loss += layer_constraint_loss

            # 对所有层的约束损失求平均
            if self._router_logits_list:
                constraint_loss /= len(self._router_logits_list)

            # 3. 将约束损失添加到总损失中
            outputs.loss += self.constraint_loss_weight * constraint_loss

        return outputs


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
