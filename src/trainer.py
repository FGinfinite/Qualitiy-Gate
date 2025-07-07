# src/trainer.py
from typing import Dict, Union, Any, Tuple

import torch
import torch.nn.functional as F
from transformers import Trainer


class CustomTrainer(Trainer):
    """
    一个自定义的 Trainer，它重写了 compute_loss 方法来包含一个专门的约束损失。
    这个约束损失旨在通过惩罚到“垃圾桶”专家的路由来鼓励模型使用“真实”的专家。
    """

    def __init__(
        self,
        *args,
        constraint_loss_weight: float,
        trash_can_loss_beta: float,
        **kwargs,
    ):
        """
        初始化 CustomTrainer。

        Args:
            *args: 传递给父类 Trainer 的位置参数。
            constraint_loss_weight (float): 应用于约束损失的权重因子。
            trash_can_loss_beta (float): 控制真实专家和垃圾桶专家之间所需 logits 差距的裕度。
            **kwargs: 传递给父类 Trainer 的关键字参数。
        """
        super().__init__(*args, **kwargs)
        self.constraint_loss_weight = constraint_loss_weight
        self.trash_can_loss_beta = trash_can_loss_beta

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        计算总损失，包括标准的交叉熵损失和自定义的约束损失。

        Args:
            model: 正在训练的模型。
            inputs: 模型的输入。
            return_outputs (bool): 是否返回模型输出和损失。

        Returns:
            总损失，或者如果 return_outputs 为 True，则返回一个包含损失和输出的元组。
        """
        # 1. 获取基础的交叉熵损失
        # 我们从模型中获取原始输出，因为它包含了 logits
        outputs = model(**inputs)
        # Trainer 的 compute_loss 期望一个包含 "loss" 和 "logits" 键的字典
        base_loss = super().compute_loss(model, inputs, return_outputs=True)[0]

        # 2. 计算约束损失
        # 从模型中访问在前向传播过程中通过挂钩捕获的 router_logits
        # .module 是为了处理 PEFT/DDP 包装器
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if not hasattr(unwrapped_model, "_router_logits_list") or not unwrapped_model._router_logits_list:
            # 如果没有 router_logits（例如，在评估期间或钩子未激活时），
            # 只返回基础损失
            return (base_loss, outputs) if return_outputs else base_loss

        all_router_logits = torch.cat(unwrapped_model._router_logits_list, dim=0)

        # 访问 MoE 层的配置以获取专家数量
        # 假设所有 MoE 层的专家数量是相同的
        first_moe_layer = unwrapped_model.model.layers[0].mlp
        original_num_experts = first_moe_layer.original_num_experts
        
        # 分离出真实专家和垃圾桶专家的 logits
        real_expert_logits = all_router_logits[:, :original_num_experts]
        trash_expert_logits = all_router_logits[:, original_num_experts:]

        # 计算每个令牌的真实专家和垃圾桶专家的平均 logits
        avg_real_expert_logits = real_expert_logits.mean(dim=1)
        avg_trash_expert_logits = trash_expert_logits.mean(dim=1)

        # 约束损失鼓励真实专家的 logits 高于垃圾桶专家的 logits
        # beta 参数控制了我们希望它们之间的差距有多大
        constraint_loss = F.relu(
            self.trash_can_loss_beta
            - (avg_real_expert_logits - avg_trash_expert_logits)
        ).mean()

        # 3. 将加权后的约束损失添加到基础损失中
        total_loss = base_loss + self.constraint_loss_weight * constraint_loss

        return (total_loss, outputs) if return_outputs else total_loss