# Copyright 2024 Select-MoE Project. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
Custom OLMoE model with trash can experts for transformers registration.
Based on the TrashCanMoE implementation in src/modeling.py.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    OlmoeSparseMoeBlock,
    load_balancing_loss_func,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SelectMoeConfig(OlmoeConfig):
    """Configuration for Select-MoE model with two-tier routing system."""

    model_type = "select_moe"

    def __init__(
        self,
        # Select-MoE specific parameters
        quality_gate_init_mean: float = 0.0,
        quality_gate_init_std: float = 0.02,
        quality_loss_weight: float = 0.01,
        trash_expert_mode: str = "zero",  # "zero", "noise", "custom"
        enable_load_balancing: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Select-MoE specific parameters  
        self.quality_gate_init_mean = quality_gate_init_mean
        self.quality_gate_init_std = quality_gate_init_std
        self.quality_loss_weight = quality_loss_weight
        self.trash_expert_mode = trash_expert_mode
        self.enable_load_balancing = enable_load_balancing

        # Ensure router logits are output by default for MoE training
        if (
            not hasattr(self, "output_router_logits")
            or self.output_router_logits is None
        ):
            self.output_router_logits = True



class QualityGate(nn.Module):
    """
    First-tier routing network for quality classification.
    Performs binary classification to determine data quality (good vs bad).
    """
    
    def __init__(self, config: SelectMoeConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.gate = nn.Linear(config.hidden_size, 2, bias=False)
        
        # Initialize gate weights
        nn.init.normal_(
            self.gate.weight.data,
            mean=config.quality_gate_init_mean,
            std=config.quality_gate_init_std,
        )
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of QualityGate.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            quality_logits: Raw logits for quality classification (batch_size, seq_len, 2)
            quality_probs: Normalized probabilities [good_ratio, bad_ratio] (batch_size, seq_len, 2)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        
        # Get quality classification logits
        quality_logits = self.gate(hidden_states_reshaped)  # (batch_size * seq_len, 2)
        
        # Convert to probabilities
        quality_probs = F.softmax(quality_logits, dim=-1)  # (batch_size * seq_len, 2)
        
        # Reshape back to original batch structure
        quality_logits = quality_logits.view(batch_size, seq_len, 2)
        quality_probs = quality_probs.view(batch_size, seq_len, 2)
        
        return quality_logits, quality_probs


class TrashExpert(nn.Module):
    """
    Trash expert that implements different output modes.
    """
    
    def __init__(self, config: SelectMoeConfig):
        super().__init__()
        self.mode = config.trash_expert_mode
        self.hidden_size = config.hidden_size
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TrashExpert.
        
        Args:
            hidden_states: Input tensor
            
        Returns:
            Output tensor based on the configured mode
        """
        if self.mode == "zero":
            return torch.zeros_like(hidden_states)
        elif self.mode == "noise":
            # Return noise with same mean and std as input
            mean = hidden_states.mean(dim=-1, keepdim=True)
            std = hidden_states.std(dim=-1, keepdim=True) + 1e-8  # Add small epsilon to avoid zero std
            noise = torch.randn_like(hidden_states) * std + mean
            return noise
        else:  # "custom" or future extensions
            return torch.zeros_like(hidden_states)


class TrashCanSparseMoeBlock(nn.Module):
    """
    A modified OlmoeSparseMoeBlock with trash can experts.
    These experts don't perform any computation and return zeros as negative incentive for the router.
    """

    def __init__(
        self, config: SelectMoeConfig, original_moe: OlmoeSparseMoeBlock = None
    ):
        super().__init__()

        # Basic MoE settings
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.original_num_experts = config.num_experts

        # Trash can experts count equals to top_k
        self.trash_can_experts_count = self.top_k

        # Total experts = original + trash can
        self.total_num_experts = (
            self.original_num_experts + self.trash_can_experts_count
        )

        # Copy experts from original MoE or create new ones
        if original_moe is not None:
            self.experts = original_moe.experts
        else:
            # Create new experts (for from_config initialization)
            from transformers.models.olmoe.modeling_olmoe import OlmoeMLP

            self.experts = nn.ModuleList(
                [OlmoeMLP(config) for _ in range(self.original_num_experts)]
            )

        # Create expanded gate
        self.gate = nn.Linear(config.hidden_size, self.total_num_experts, bias=False)

        # Initialize gate weights
        if original_moe is not None:
            # Copy original gate weights
            self.gate.weight.data[: self.original_num_experts, :] = (
                original_moe.gate.weight.data.to(self.gate.weight.dtype)
            )
        else:
            # Initialize original expert weights normally
            nn.init.normal_(
                self.gate.weight.data[: self.original_num_experts, :],
                mean=0.0,
                std=config.initializer_range,
            )

        # Initialize trash can expert weights
        nn.init.normal_(
            self.gate.weight.data[self.original_num_experts :, :],
            mean=config.trash_can_init_mean,
            std=config.trash_can_init_std,
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of TrashCanSparseMoeBlock.

        Routes tokens to original experts or trash cans, where they are effectively zeroed out.
        Always returns router_logits for loss computation.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        # Get routing logits from expanded gate
        router_logits = self.gate(hidden_states_reshaped)

        # Get top-k routing weights and selected experts
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        routing_weights = routing_weights.to(hidden_states.dtype)

        # Initialize final hidden states as zeros
        # Tokens routed to trash cans will remain zero
        final_hidden_states = torch.zeros_like(hidden_states_reshaped)

        # Create one-hot mask to identify which tokens are routed to which expert
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.total_num_experts
        ).permute(2, 1, 0)

        # Process only original experts (trash can experts output zeros by default)
        for expert_idx in range(self.original_num_experts):
            expert_layer = self.experts[expert_idx]
            # Find which tokens are routed to current expert
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # Select hidden states for current expert
            current_state = hidden_states_reshaped[None, top_x].reshape(-1, hidden_dim)

            # Compute expert output and scale by routing weights
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )

            # Add expert output to final hidden states
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )

        # Always return router_logits for loss computation
        return final_hidden_states, router_logits


class SelectMoeDecoderLayer(OlmoeDecoderLayer):
    """Decoder layer with two-tier routing system."""

    def __init__(self, config: SelectMoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        # Add our two-tier routing components on top of existing MLP
        # First-tier: Quality gate for good/bad classification
        self.quality_gate = QualityGate(config)
        
        # Second-tier: Reuse the existing MoE block (self.mlp) as normal_moe
        self.normal_moe = self.mlp
        # Remove the original mlp reference to avoid shared tensors issue
        delattr(self, 'mlp')
        
        # Trash expert for low-quality data processing
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
        Forward pass with two-tier routing system.
        
        Args:
            hidden_states: Input tensor
            ... (other standard transformer layer arguments)
            
        Returns:
            tuple containing:
            - output hidden states
            - present key value (if use_cache)
            - attention weights (if output_attentions)  
            - router logits including quality_logits and moe_logits (if output_router_logits)
        """
        residual = hidden_states

        # Self-attention computation (inherited from parent class)
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self attention
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

        # MLP computation with two-tier routing
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # First-tier routing: Quality classification
        quality_logits, quality_probs = self.quality_gate(hidden_states)
        good_ratio = quality_probs[..., 0:1]  # Shape: (batch_size, seq_len, 1)
        bad_ratio = quality_probs[..., 1:2]   # Shape: (batch_size, seq_len, 1)
        
        # Second-tier routing: Normal MoE processing
        y_normal, moe_router_logits = self.normal_moe(hidden_states)
        
        # Trash expert processing
        y_trash = self.trash_expert(hidden_states)
        
        # Combine outputs: y = good_ratio * y_normal + bad_ratio * y_trash
        hidden_states = good_ratio * y_normal + bad_ratio * y_trash
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            # Return both quality logits and MoE router logits
            router_logits = {
                'quality_logits': quality_logits,
                'moe_logits': moe_router_logits
            }
            outputs += (router_logits,)

        return outputs


class SelectMoePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
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
    Transformer decoder consisting of *config.num_hidden_layers* layers with TrashCanMoE.
    """

    def __init__(self, config: SelectMoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                SelectMoeDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # Import and use the same components as OlmoeModel
        from transformers.models.olmoe.modeling_olmoe import (
            OlmoeRMSNorm,
            OlmoeRotaryEmbedding,
        )

        self.norm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = OlmoeRotaryEmbedding(config=config)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
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
        # Use the same implementation as OlmoeModel but with our custom layers
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Simple implementation for cache and attention mask handling
        if cache_position is None:
            cache_position = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Create causal attention mask
        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
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

        # add hidden states from the last decoder layer
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
        """Simple implementation of causal mask update."""
        # For simplicity, return None to let the attention layers handle causal masking internally
        return None


def quality_classification_loss(
    router_logits: List[dict], config: SelectMoeConfig, debug: bool = False
) -> torch.Tensor:
    """
    Compute quality classification loss using sigmoid(good_ratio).
    
    This loss encourages the model to reduce good_ratio values, forcing it to learn
    to distinguish between good and bad quality data.
    
    Args:
        router_logits: List of dictionaries containing 'quality_logits' and 'moe_logits' for each layer
        config: Model configuration
        debug: Whether to print debug information
        
    Returns:
        Quality classification loss tensor
    """
    if debug:
        print(f"\n=== Quality Classification Loss Debug ===")
        print(f"router_logits type: {type(router_logits)}")
        print(f"Number of layers (len(router_logits)): {len(router_logits)}")

    if len(router_logits) == 0:
        return torch.tensor(0.0, device="cpu")

    total_loss = 0.0
    num_layers = 0

    for layer_idx, layer_router_logits in enumerate(router_logits):
        if layer_router_logits is None or 'quality_logits' not in layer_router_logits:
            continue

        quality_logits = layer_router_logits['quality_logits']  # Shape: (batch_size, seq_len, 2)
        
        if debug:
            print(f"\n--- Layer {layer_idx} ---")
            print(f"quality_logits shape: {quality_logits.shape}")
            print(f"quality_logits device: {quality_logits.device}")

        # Get quality probabilities
        quality_probs = F.softmax(quality_logits, dim=-1)  # Shape: (batch_size, seq_len, 2)
        good_ratio = quality_probs[..., 0]  # Shape: (batch_size, seq_len)
        
        if debug:
            print(f"good_ratio shape: {good_ratio.shape}")
            print(f"good_ratio min/max/mean: {good_ratio.min().item():.4f}/{good_ratio.max().item():.4f}/{good_ratio.mean().item():.4f}")

        # Apply sigmoid to good_ratio and use as loss
        # This encourages the model to reduce good_ratio (i.e., classify more data as bad quality)
        layer_loss = torch.sigmoid(good_ratio).mean()
        
        if debug:
            print(f"layer_loss: {layer_loss.item():.6f}")

        total_loss += layer_loss
        num_layers += 1

    final_loss = total_loss / max(num_layers, 1)
    
    if debug:
        print(f"\nFinal quality classification loss: {final_loss.item():.6f}")
        print("=== End Debug ===\n")

    return final_loss


class SelectMoeForCausalLM(SelectMoePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = SelectMoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        # Note: total experts include trash can experts for load balancing
        self.num_experts = (
            config.num_experts + config.num_experts_per_tok
        )  # trash_can_experts_count
        self.num_experts_per_tok = config.num_experts_per_tok

        # Initialize weights and apply final processing
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

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        # print(f"Loss: {loss.item() if loss is not None else 'N/A'}")

        aux_loss = None
        if output_router_logits:
            # Load balancing loss (optional)
            if self.config.enable_load_balancing:
                aux_loss = load_balancing_loss_func(
                    outputs.router_logits if return_dict else outputs[-1],
                    self.num_experts,
                    self.num_experts_per_tok,
                    attention_mask,
                )
            else:
                aux_loss = torch.tensor(0.0, device=hidden_states.device)

            # Quality classification loss (always computed)
            quality_loss = quality_classification_loss(
                outputs.router_logits if return_dict else outputs[-1], self.config
            )
            aux_loss += self.config.quality_loss_weight * quality_loss

            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        # print(f"Router Aux Loss: {aux_loss.item() if aux_loss is not None else 'N/A'}")
        # print(f"Total Loss: {loss.item() if loss is not None else 'N/A'}")

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
        Prepare inputs for generation. This method is required by PEFT.
        """
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif (
                input_ids.shape[1] != cache_position.shape[0]
            ):  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
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


def replace_moe_layers_with_trashcan(model: nn.Module, config: SelectMoeConfig):
    """
    Recursively traverse the model and replace all OlmoeSparseMoeBlock instances with TrashCanSparseMoeBlock.
    This function preserves pretrained weights.
    """
    for name, module in model.named_children():
        if isinstance(module, OlmoeSparseMoeBlock):
            # Create a new TrashCanSparseMoeBlock to replace the original MoE block
            new_moe = TrashCanSparseMoeBlock(config, module)
            setattr(model, name, new_moe)
        elif len(list(module.children())) > 0:
            # Recurse into submodules
            replace_moe_layers_with_trashcan(module, config)


def register_select_moe():
    """Register Select-MoE model for AutoConfig and AutoModel."""
    AutoConfig.register("select_moe", SelectMoeConfig)
    AutoModel.register(SelectMoeConfig, SelectMoeModel)
    AutoModelForCausalLM.register(SelectMoeConfig, SelectMoeForCausalLM)


__all__ = [
    "SelectMoeConfig",
    "SelectMoeModel",
    "SelectMoeForCausalLM",
    "TrashCanSparseMoeBlock",
    "replace_moe_layers_with_trashcan",
    "register_select_moe",
]
