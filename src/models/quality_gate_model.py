# Copyright 2024 Quality-Gate Project. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
质量门控模型：基于Qwen3-1.7B，在FFN层之前插入质量门控
用于数据质量评估和筛选
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
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3DecoderLayer,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


class QualityGateConfig(Qwen3Config):
    """质量门控模型配置，扩展自Qwen3Config"""

    model_type = "quality_gate"

    def __init__(
        self,
        # 质量门控特定参数
        quality_gate_init_mean: float = 0.0,
        quality_gate_init_std: float = 0.02,
        quality_loss_weight: float = 0.5,
        # 质量损失配置
        quality_loss_type: str = "linear",  # 质量损失类型（仅支持linear）
        # 损失平均策略
        sample_wise_averaging: bool = True,
        # 调试配置
        quality_loss_debug: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # 质量门控特定参数
        self.quality_gate_init_mean = quality_gate_init_mean
        self.quality_gate_init_std = quality_gate_init_std
        self.quality_loss_weight = quality_loss_weight

        # 质量损失配置
        self.quality_loss_type = quality_loss_type
        self.sample_wise_averaging = sample_wise_averaging
        self.quality_loss_debug = quality_loss_debug


class QualityGate(nn.Module):
    """
    质量分类的门控网络
    输出单个分数，通过sigmoid得到good_ratio
    """

    def __init__(self, config: QualityGateConfig):
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
        QualityGate的前向传播

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


class QualityGateDecoderLayer(Qwen3DecoderLayer):
    """带有质量门控的解码器层"""

    def __init__(self, config: QualityGateConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        # 在FFN之前添加质量门控
        self.quality_gate = QualityGate(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        """
        使用质量门控的前向传播

        Args:
            hidden_states: 输入张量
            ... (其他标准transformer层参数)

        Returns:
            包含以下内容的元组:
            - 输出隐藏状态
            - 注意力权重(如果output_attentions)
            - 质量门控logits(如果output_router_logits)
        """
        residual = hidden_states

        # 自注意力层
        hidden_states = self.input_layernorm(hidden_states)

        # 自注意力 (Qwen3返回值不包含present_key_value)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # FFN层
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # 质量门控
        quality_score, good_ratio = self.quality_gate(hidden_states)

        # FFN处理
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            # 返回质量分数
            outputs += (quality_score,)

        return outputs


class QualityGatePreTrainedModel(PreTrainedModel):
    """
    处理权重初始化和提供下载加载预训练模型简单接口的抽象类
    """

    config_class = QualityGateConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["QualityGateDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

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


class QualityGateModel(QualityGatePreTrainedModel):
    """
    由*config.num_hidden_layers*层组成的Transformer解码器，使用质量门控架构
    """

    def __init__(self, config: QualityGateConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([QualityGateDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

        # 使用Qwen3的RMSNorm和RotaryEmbedding
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3RMSNorm,
            Qwen3RotaryEmbedding,
        )

        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # 检查是否有sliding window layers
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types if hasattr(self.config, "layer_types") else False

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
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = output_router_logits if output_router_logits is not None else False
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

        # 处理cache position (Qwen3风格)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # 创建因果注意力掩码 (Qwen3风格)
        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # 创建在解码器层之间共享的位置嵌入
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 解码器层
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask_mapping[decoder_layer.attention_type],
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            # Qwen3不在layer输出中返回cache，cache在model级别处理
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                # 质量门控logits在最后
                if output_attentions:
                    all_router_logits += (layer_outputs[2],)
                else:
                    all_router_logits += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # 添加最后一个解码器层的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Qwen3使用past_key_values而不是next_decoder_cache
        next_cache = past_key_values if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_router_logits,
                ]
                if v is not None
            )

        # 使用自定义输出类
        return QualityGateModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class QualityGateModelOutput:
    """质量门控模型输出"""

    def __init__(
        self,
        last_hidden_state: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        hidden_states: Optional[Tuple[torch.Tensor]] = None,
        attentions: Optional[Tuple[torch.Tensor]] = None,
        router_logits: Optional[Tuple[torch.Tensor]] = None,
    ):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.router_logits = router_logits


def quality_classification_loss(
    router_logits: List[torch.Tensor],
    config: QualityGateConfig,
    attention_mask: Optional[torch.Tensor] = None,
    debug: bool = False,
) -> torch.Tensor:
    """
    计算质量分类损失

    使用线性损失：直接使用sigmoid后的good_ratio作为损失，鼓励降低good_ratio

    Args:
        router_logits: 包含每层'quality_score'的张量列表
        config: 模型配置
        attention_mask: 注意力掩码，用于排除padding tokens (batch_size, seq_len)
        debug: 是否打印调试信息

    Returns:
        质量分类损失张量
    """
    if debug:
        print("\n=== 质量分类损失调试 ===")
        print(f"router_logits类型: {type(router_logits)}")
        print(f"层数量: {len(router_logits)}")

    if len(router_logits) == 0:
        return torch.tensor(0.0, device="cpu")

    total_loss = 0.0
    num_layers = 0

    for layer_idx, quality_score in enumerate(router_logits):
        if quality_score is None:
            continue

        # quality_score 形状: (batch_size, seq_len, 1)
        if debug and layer_idx % 5 == 0:
            print(f"\n--- 第{layer_idx}层 ---")
            print(f"quality_score形状: {quality_score.shape}")
            print(f"quality_score设备: {quality_score.device}")

        # 计算good_ratio
        good_ratio = torch.sigmoid(quality_score)  # 形状: (batch_size, seq_len, 1)

        if debug and layer_idx % 5 == 0:
            print(f"good_ratio 最小/最大/均值: {good_ratio.min().item():.4f}/{good_ratio.max().item():.4f}/{good_ratio.mean().item():.4f}")

        # 线性损失：直接使用good_ratio作为损失
        layer_loss_raw = good_ratio.squeeze(-1)  # (batch_size, seq_len)

        # 应用attention mask排除padding tokens
        if attention_mask is not None:
            if attention_mask.shape != layer_loss_raw.shape:
                attention_mask_expanded = attention_mask
            else:
                attention_mask_expanded = attention_mask

            masked_loss = layer_loss_raw * attention_mask_expanded.float()

            # 根据配置选择平均化策略
            if hasattr(config, "sample_wise_averaging") and config.sample_wise_averaging:
                # Sample-wise averaging: 先计算每个样本的平均损失，再对样本求平均
                batch_size = layer_loss_raw.shape[0]
                sample_losses = []

                for i in range(batch_size):
                    sample_mask = attention_mask_expanded[i]
                    sample_loss = masked_loss[i]
                    valid_tokens_in_sample = sample_mask.sum()

                    if valid_tokens_in_sample > 0:
                        sample_avg_loss = sample_loss.sum() / valid_tokens_in_sample
                        sample_losses.append(sample_avg_loss)

                if sample_losses:
                    layer_loss = torch.stack(sample_losses).mean()
                else:
                    layer_loss = torch.tensor(0.0, device=quality_score.device)

                if debug and layer_idx % 5 == 0:
                    print(f"使用Sample-wise平均化策略，有效样本数量: {len(sample_losses)}")
            else:
                # Token-wise averaging: 直接对所有有效token求和平均
                valid_tokens = attention_mask_expanded.sum()

                if valid_tokens > 0:
                    layer_loss = masked_loss.sum() / valid_tokens
                else:
                    layer_loss = torch.tensor(0.0, device=quality_score.device)

                if debug and layer_idx % 5 == 0:
                    print(f"使用Token-wise平均化策略，有效token数量: {valid_tokens.item()}")

        else:
            layer_loss = layer_loss_raw.mean()

        if debug and layer_idx % 5 == 0:
            print(f"层损失: {layer_loss.item():.6f}")

        total_loss += layer_loss
        num_layers += 1

    final_loss = total_loss / max(num_layers, 1)

    if debug:
        print(f"\n最终质量分类损失: {final_loss.item():.6f}")
        print("=== 调试结束 ===\n")

    return final_loss


class QualityGateForCausalLM(QualityGatePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = QualityGateModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def loss_function(self, logits, labels, vocab_size, **kwargs):
        """
        重写损失函数以正确处理填充token
        """
        # 移位logits和labels用于因果语言建模
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 展平token以计算损失
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # 使用ignore_index=-100忽略填充token
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits, shift_labels)

        return loss

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
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        if output_router_logits is None:
            output_router_logits = self.training
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs
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

        hidden_states = outputs.last_hidden_state if return_dict else outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        # 质量损失
        aux_loss = None
        if output_router_logits and outputs.router_logits is not None:
            quality_loss = quality_classification_loss(
                outputs.router_logits,
                self.config,
                attention_mask=attention_mask,
                debug=self.config.quality_loss_debug,
            )

            aux_loss = self.config.quality_loss_weight * quality_loss

            if labels is not None and loss is not None:
                loss += aux_loss.to(loss.device)

        if not return_dict:
            output = (logits,) + (outputs[1:] if return_dict else outputs[1:])
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        # 注意：使用 past_key_values 字段传递 router_logits（HuggingFace的常见做法）
        # 因为 CausalLMOutputWithPast 不原生支持 router_logits
        return_output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if return_dict else None,
            hidden_states=outputs.hidden_states if return_dict else None,
            attentions=outputs.attentions if return_dict else None,
        )

        # 添加 router_logits 作为额外属性（用于数据选择阶段）
        if output_router_logits and hasattr(outputs, "router_logits"):
            return_output.router_logits = outputs.router_logits

        return return_output

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
        """为生成准备输入"""
        if past_key_values is not None:
            if inputs_embeds is not None:
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

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


def register_quality_gate():
    """为AutoConfig和AutoModel注册质量门控模型"""
    AutoConfig.register("quality_gate", QualityGateConfig)
    AutoModel.register(QualityGateConfig, QualityGateModel)
    AutoModelForCausalLM.register(QualityGateConfig, QualityGateForCausalLM)


__all__ = [
    "QualityGateConfig",
    "QualityGateModel",
    "QualityGateForCausalLM",
    "register_quality_gate",
]
