# Copyright 2024 Select-MoE Project. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
使用两层路由架构的自定义Select-MoE模型，用于transformers注册。
基于质量门控和MoE专家的组合实现。
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
)
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.models.olmoe.modeling_olmoe import (
    OlmoeConfig,
    OlmoeDecoderLayer,
    load_balancing_loss_func,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SelectMoeConfig(OlmoeConfig):
    """Select-MoE模型配置，带有两层路由系统。"""

    model_type = "select_moe"

    def __init__(
        self,
        # Select-MoE specific parameters
        quality_gate_init_mean: float = 0.0,
        quality_gate_init_std: float = 0.02,
        quality_loss_weight: float = 0.5,
        trash_expert_mode: str = "zero",  # "zero", "noise", "custom"
        enable_load_balancing: bool = False,
        # Quality loss configuration
        quality_loss_type: str = "sigmoid",  # "sigmoid", "beta_moment_matching", "mean_variance_regularization"
        # Beta moment matching parameters
        beta_target_mean: float = 0.5,
        beta_target_var: float = 0.05,
        w_mean: float = 1.0,
        w_var: float = 1.0,
        # Mean-variance regularization parameters
        lambda_var: float = 0.1,
        # Debug configuration
        quality_loss_debug: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Select-MoE特定参数
        self.quality_gate_init_mean = quality_gate_init_mean
        self.quality_gate_init_std = quality_gate_init_std
        self.quality_loss_weight = quality_loss_weight
        self.trash_expert_mode = trash_expert_mode
        self.enable_load_balancing = enable_load_balancing

        # Quality loss configuration
        self.quality_loss_type = quality_loss_type
        self.beta_target_mean = beta_target_mean
        self.beta_target_var = beta_target_var
        self.w_mean = w_mean
        self.w_var = w_var
        self.lambda_var = lambda_var
        self.quality_loss_debug = quality_loss_debug

        # 确保默认输出router logits用于MoE训练
        if not hasattr(self, "output_router_logits") or self.output_router_logits is None:
            self.output_router_logits = True


class QualityGate(nn.Module):
    """
    质量分类的第一层路由网络。
    输出单个分数，通过sigmoid得到good_ratio。
    """

    def __init__(self, config: SelectMoeConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.gate = nn.Linear(config.hidden_size, 1, bias=False)

        # 初始化门控权重
        nn.init.normal_(
            self.gate.weight.data,
            mean=config.quality_gate_init_mean,
            std=config.quality_gate_init_std,
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        QualityGate的前向传播。

        Args:
            hidden_states: 输入张量，形状为(batch_size, seq_len, hidden_size)

        Returns:
            quality_score: 质量分数的原始值 (batch_size, seq_len, 1)
            good_ratio: sigmoid后的good比率 (batch_size, seq_len, 1)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        # 获取质量分数 (单个值)
        quality_score = self.gate(hidden_states_reshaped)  # (batch_size * seq_len, 1)

        # 通过sigmoid得到good_ratio
        good_ratio = torch.sigmoid(quality_score)  # (batch_size * seq_len, 1)

        # 重新整形回原始批次结构
        quality_score = quality_score.view(batch_size, seq_len, 1)
        good_ratio = good_ratio.view(batch_size, seq_len, 1)

        return quality_score, good_ratio


class TrashExpert(nn.Module):
    """
    实现不同输出模式的垃圾专家。
    """

    def __init__(self, config: SelectMoeConfig):
        super().__init__()
        self.mode = config.trash_expert_mode
        self.hidden_size = config.hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        TrashExpert的前向传播。

        Args:
            hidden_states: 输入张量

        Returns:
            基于配置模式的输出张量
        """
        if self.mode == "zero":
            return torch.zeros_like(hidden_states)
        elif self.mode == "noise":
            # 返回与输入具有相同均值和标准差的噪声
            mean = hidden_states.mean(dim=-1, keepdim=True)
            std = hidden_states.std(dim=-1, keepdim=True) + 1e-8  # 添加小epsilon避免零标准差
            noise = torch.randn_like(hidden_states) * std + mean
            return noise
        else:  # "custom"或未来扩展
            return torch.zeros_like(hidden_states)


class SelectMoeDecoderLayer(OlmoeDecoderLayer):
    """带有两层路由系统的解码器层。"""

    def __init__(self, config: SelectMoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        # 在现有MLP基础上添加我们的两层路由组件
        # 第一层：用于好/坏分类的质量门控
        self.quality_gate = QualityGate(config)

        # 第二层：重用现有的MoE块(self.mlp)作为normal_moe
        self.normal_moe = self.mlp
        # 移除原始mlp引用以避免共享张量问题
        delattr(self, "mlp")

        # 用于低质量数据处理的垃圾专家
        self.trash_expert = TrashExpert(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        使用两层路由系统的前向传播。

        Args:
            hidden_states: 输入张量
            ... (其他标准transformer层参数)

        Returns:
            包含以下内容的元组:
            - 输出隐藏状态
            - 当前键值(如果use_cache)
            - 注意力权重(如果output_attentions)
            - 包括quality_logits和moe_logits的router logits(如果output_router_logits)
        """
        residual = hidden_states

        # 自注意力计算（从父类继承）
        hidden_states = self.input_layernorm(hidden_states)

        # 自注意力
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # 使用两层路由的MLP计算
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # 第一层路由：质量分类
        quality_score, good_ratio = self.quality_gate(hidden_states)
        bad_ratio = 1.0 - good_ratio  # 形状: (batch_size, seq_len, 1)

        # 第二层路由：正常MoE处理
        y_normal, moe_router_logits = self.normal_moe(hidden_states)

        # 垃圾专家处理
        y_trash = self.trash_expert(hidden_states)

        # 组合输出: y = good_ratio * y_normal + bad_ratio * y_trash
        hidden_states = good_ratio * y_normal + bad_ratio * y_trash
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            # 返回质量分数和MoE路由器logits
            router_logits = {"quality_score": quality_score, "moe_logits": moe_router_logits}
            outputs += (router_logits,)

        return outputs


class SelectMoePreTrainedModel(PreTrainedModel):
    """
    处理权重初始化和提供下载加载预训练模型简单接口的抽象类。
    """

    config_class = SelectMoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SelectMoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class SelectMoeModel(SelectMoePreTrainedModel):
    """
    由*config.num_hidden_layers*层组成的Transformer解码器，使用两层路由架构。
    """

    def __init__(self, config: SelectMoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([SelectMoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

        # 导入并使用与OlmoeModel相同的组件
        from transformers.models.olmoe.modeling_olmoe import (
            OlmoeRMSNorm,
            OlmoeRotaryEmbedding,
        )

        self.norm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = OlmoeRotaryEmbedding(config=config)

        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        # 使用与OlmoeModel相同的实现，但使用我们的自定义层
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 缓存和注意力掩码处理的简单实现
        if cache_position is None:
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # 创建因果注意力掩码
        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        # 嵌入位置
        hidden_states = inputs_embeds

        # 创建在解码器层之间共享的位置嵌入
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 解码器层
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # 添加最后一个解码器层的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values,
        output_attentions: bool,
    ):
        """因果掩码更新的简单实现。"""
        # 为简单起见，返回None让注意力层内部处理因果掩码
        return None


def beta_moment_matching_loss(
    good_ratio: torch.Tensor,
    attention_mask: torch.Tensor,
    target_mean: float = 0.5,
    target_var: float = 0.05,
    w_mean: float = 1.0,
    w_var: float = 1.0,
) -> torch.Tensor:
    """
    方案一：矩匹配损失 (基于Beta分布)
    将good_ratio分布的矩与目标Beta分布的矩进行匹配。

    Args:
        good_ratio: Good ratio tensor, shape (batch_size, seq_len, 1)
        attention_mask: Attention mask for excluding padding tokens
        target_mean: Target mean for Beta distribution
        target_var: Target variance for Beta distribution
        w_mean: Weight for mean loss component
        w_var: Weight for variance loss component

    Returns:
        Loss tensor with same shape as good_ratio
    """
    good_ratio_squeezed = good_ratio.squeeze(-1)

    if attention_mask is not None:
        if attention_mask.shape != good_ratio_squeezed.shape:
            attention_mask = attention_mask.expand_as(good_ratio_squeezed)
        valid_ratios = torch.masked_select(good_ratio_squeezed, attention_mask.bool())
        if valid_ratios.numel() == 0:
            return torch.zeros_like(good_ratio_squeezed)
    else:
        valid_ratios = good_ratio_squeezed.flatten()

    batch_mean = valid_ratios.mean()
    batch_var = valid_ratios.var()

    loss_mean = (batch_mean - target_mean) ** 2
    loss_var = (batch_var - target_var) ** 2

    batch_loss = w_mean * loss_mean + w_var * loss_var

    return torch.full_like(good_ratio_squeezed, batch_loss.item())


def mean_variance_regularization_loss(
    good_ratio: torch.Tensor,
    attention_mask: torch.Tensor,
    lambda_var: float = 0.1,
) -> torch.Tensor:
    """
    方案二：均值-方差正则化
    将均值拉向0.5并鼓励方差最大化。

    Args:
        good_ratio: Good ratio tensor, shape (batch_size, seq_len, 1)
        attention_mask: Attention mask for excluding padding tokens
        lambda_var: Weight for variance regularization term

    Returns:
        Loss tensor with same shape as good_ratio
    """
    good_ratio_squeezed = good_ratio.squeeze(-1)

    if attention_mask is not None:
        if attention_mask.shape != good_ratio_squeezed.shape:
            attention_mask = attention_mask.expand_as(good_ratio_squeezed)
        valid_ratios = torch.masked_select(good_ratio_squeezed, attention_mask.bool())
        if valid_ratios.numel() == 0:
            return torch.zeros_like(good_ratio_squeezed)
    else:
        valid_ratios = good_ratio_squeezed.flatten()

    batch_mean = valid_ratios.mean()
    batch_var = valid_ratios.var()

    loss_mean = (batch_mean - 0.5) ** 2
    loss_var = -lambda_var * batch_var

    batch_loss = loss_mean + loss_var

    return torch.full_like(good_ratio_squeezed, batch_loss.item())


def quality_classification_loss(
    router_logits: List[dict],
    config: SelectMoeConfig,
    attention_mask: Optional[torch.Tensor] = None,
    loss_type: str = "sigmoid",
    custom_loss_fn: Optional[callable] = None,
    debug: bool = False,
) -> torch.Tensor:
    """
    使用可扩展的损失函数计算质量分类损失。

    该损失支持多种计算方式，并正确处理padding tokens。

    Args:
        router_logits: 包含每层'quality_score'和'moe_logits'的字典列表
        config: 模型配置
        attention_mask: 注意力掩码，用于排除padding tokens (batch_size, seq_len)
        loss_type: 损失类型 ("sigmoid", "mse", "custom")
        custom_loss_fn: 自定义损失函数，接受(good_ratio, attention_mask)返回loss
        debug: 是否打印调试信息

    Returns:
        质量分类损失张量
    """
    if debug:
        print("\n=== 质量分类损失调试 ===")
        print(f"router_logits类型: {type(router_logits)}")
        print(f"层数量 (len(router_logits)): {len(router_logits)}")
        print(f"loss_type: {loss_type}")

    if len(router_logits) == 0:
        return torch.tensor(0.0, device="cpu")

    total_loss = 0.0
    num_layers = 0

    for layer_idx, layer_router_logits in enumerate(router_logits):
        if layer_router_logits is None or "quality_score" not in layer_router_logits:
            continue

        quality_score = layer_router_logits["quality_score"]  # Shape: (batch_size, seq_len, 1)

        if debug and layer_idx % 5 == 0:
            print(f"\n--- 第{layer_idx}层 ---")
            print(f"quality_score形状: {quality_score.shape}")
            print(f"quality_score设备: {quality_score.device}")

        # 计算good_ratio
        good_ratio = torch.sigmoid(quality_score)  # 形状: (batch_size, seq_len, 1)

        if debug:
            print(f"good_ratio形状: {good_ratio.shape}")
            print(f"good_ratio 最小/最大/均值: {good_ratio.min().item():.4f}/{good_ratio.max().item():.4f}/{good_ratio.mean().item():.4f}")

        # 计算层损失
        if loss_type == "sigmoid":
            # 直接使用good_ratio作为损失，鼓励降低good_ratio
            layer_loss_raw = good_ratio.squeeze(-1)  # (batch_size, seq_len)
        elif loss_type == "mse":
            # MSE损失，目标是将good_ratio推向0
            target = torch.zeros_like(good_ratio.squeeze(-1))
            layer_loss_raw = (good_ratio.squeeze(-1) - target) ** 2
        elif loss_type == "beta_moment_matching":
            # 使用矩匹配损失 (方案一)
            layer_loss_raw = beta_moment_matching_loss(
                good_ratio, attention_mask, target_mean=config.beta_target_mean, target_var=config.beta_target_var, w_mean=config.w_mean, w_var=config.w_var
            )
        elif loss_type == "mean_variance_regularization":
            # 使用均值-方差正则化损失 (方案二)
            layer_loss_raw = mean_variance_regularization_loss(good_ratio, attention_mask, lambda_var=config.lambda_var)
        elif loss_type == "custom" and custom_loss_fn is not None:
            # 使用自定义损失函数
            layer_loss_raw = custom_loss_fn(good_ratio, attention_mask)
            if layer_loss_raw.dim() > 2:
                layer_loss_raw = layer_loss_raw.squeeze(-1)
        else:
            # 默认回退到sigmoid
            layer_loss_raw = good_ratio.squeeze(-1)

        # 应用attention mask排除padding tokens
        if attention_mask is not None:
            # 确保attention_mask形状匹配
            if attention_mask.shape != layer_loss_raw.shape:
                if debug:
                    print(f"注意: attention_mask形状 {attention_mask.shape} != loss形状 {layer_loss_raw.shape}")
                # 如果形状不匹配，广播或调整
                attention_mask_expanded = attention_mask
                if attention_mask.dim() == 2 and layer_loss_raw.dim() == 2:
                    attention_mask_expanded = attention_mask
                elif attention_mask.dim() == 2 and layer_loss_raw.dim() == 3:
                    attention_mask_expanded = attention_mask.unsqueeze(-1)
            else:
                attention_mask_expanded = attention_mask

            # 只计算有效token的损失
            masked_loss = layer_loss_raw * attention_mask_expanded.float()
            valid_tokens = attention_mask_expanded.sum()

            if valid_tokens > 0:
                layer_loss = masked_loss.sum() / valid_tokens
            else:
                layer_loss = torch.tensor(0.0, device=quality_score.device)

            if debug:
                print(f"有效token数量: {valid_tokens.item()}")
        else:
            # 没有attention_mask时，对所有位置求平均
            layer_loss = layer_loss_raw.mean()

        if debug:
            print(f"层损失: {layer_loss.item():.6f}")

        total_loss += layer_loss
        num_layers += 1

    final_loss = total_loss / max(num_layers, 1)

    if debug:
        print(f"\n最终质量分类损失: {final_loss.item():.6f}")
        print("=== 调试结束 ===\n")

    return final_loss


class SelectMoeForCausalLM(SelectMoePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = SelectMoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        # MoE专家总数（用于负载均衡）
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        # if input_ids is not None:
        #     print(f"[SelectMoeForCausalLM] seq_len: {input_ids.shape[1]}")
        # elif inputs_embeds is not None:
        #     print(f"[SelectMoeForCausalLM] seq_len: {inputs_embeds.shape[1]}")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        # print(f"Loss: {loss.item() if loss is not None else 'N/A'}")

        aux_loss = None
        if output_router_logits:
            # Load balancing loss (optional)
            if self.config.enable_load_balancing:
                # 提取MoE router logits进行负载均衡
                moe_router_logits = []
                router_logits_data = outputs.router_logits if return_dict else outputs[-1]
                for layer_logits in router_logits_data:
                    if layer_logits is not None and "moe_logits" in layer_logits:
                        moe_router_logits.append(layer_logits["moe_logits"])

                if moe_router_logits:
                    aux_loss = load_balancing_loss_func(
                        moe_router_logits,
                        self.num_experts,
                        self.num_experts_per_tok,
                        attention_mask,
                    )
                else:
                    aux_loss = torch.tensor(0.0, device=hidden_states.device)
            else:
                aux_loss = torch.tensor(0.0, device=hidden_states.device)

            # Quality classification loss (always computed)
            quality_loss = quality_classification_loss(
                outputs.router_logits if return_dict else outputs[-1],
                self.config,
                attention_mask=attention_mask,
                loss_type=self.config.quality_loss_type,
                debug=self.config.quality_loss_debug,
            )

            if self.config.quality_loss_debug:
                print(f"Quality Loss: {quality_loss.item()}")
            aux_loss += self.config.quality_loss_weight * quality_loss

            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        if self.config.quality_loss_debug:
            print(f"Router Aux Loss: {aux_loss.item() if aux_loss is not None else 'N/A'}")
            print(f"Total Loss: {loss.item() if loss is not None else 'N/A'}")

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        """
        为生成准备输入。PEFT需要此方法。
        """
        # 如果有缓存：通过`cache_position`切片`input_ids`，只保留未处理的tokens
        # 异常1：传递input_embeds时，input_ids可能缺少条目
        # 异常2：一些生成方法对input_ids进行特殊切片，所以我们不需要在这里做
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # 为批次生成动态创建position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 如果传递了`inputs_embeds`，我们只想在第一个生成步骤中使用它们
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


def register_select_moe():
    """为AutoConfig和AutoModel注册Select-MoE模型。"""
    AutoConfig.register("select_moe", SelectMoeConfig)
    AutoModel.register(SelectMoeConfig, SelectMoeModel)
    AutoModelForCausalLM.register(SelectMoeConfig, SelectMoeForCausalLM)


__all__ = [
    "SelectMoeConfig",
    "SelectMoeModel",
    "SelectMoeForCausalLM",
    "register_select_moe",
]
