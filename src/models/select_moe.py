# Copyright 2024 Select-MoE Project. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
Custom OLMoE model with trash can experts for transformers registration.
Based on the TrashCanMoE implementation in src/modeling.py.
"""

from typing import Optional, Union, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
)
from transformers.models.olmoe.modeling_olmoe import (
    OlmoeConfig,
    OlmoeModel,
    OlmoeForCausalLM,
    OlmoeSparseMoeBlock,
    OlmoeDecoderLayer,
    load_balancing_loss_func,
)
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SelectMoeConfig(OlmoeConfig):
    """Configuration for Select-MoE model with trash can experts."""
    
    model_type = "select_moe"
    
    def __init__(
        self,
        # Select-MoE specific parameters
        trash_can_init_mean: float = 0.0,
        trash_can_init_std: float = 0.02,
        constraint_loss_weight: float = 0.01,
        trash_can_loss_alpha: float = 1.0,
        trash_can_loss_beta: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Select-MoE specific parameters
        self.trash_can_init_mean = trash_can_init_mean
        self.trash_can_init_std = trash_can_init_std
        self.constraint_loss_weight = constraint_loss_weight
        self.trash_can_loss_alpha = trash_can_loss_alpha
        self.trash_can_loss_beta = trash_can_loss_beta


class TrashCanSparseMoeBlock(nn.Module):
    """
    A modified OlmoeSparseMoeBlock with trash can experts.
    These experts don't perform any computation and return zeros as negative incentive for the router.
    """
    
    def __init__(self, config: SelectMoeConfig, original_moe: OlmoeSparseMoeBlock = None):
        super().__init__()
        
        # Basic MoE settings
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.original_num_experts = config.num_experts
        
        # Trash can experts count equals to top_k
        self.trash_can_experts_count = self.top_k
        
        # Total experts = original + trash can
        self.total_num_experts = self.original_num_experts + self.trash_can_experts_count
        
        # Copy experts from original MoE or create new ones
        if original_moe is not None:
            self.experts = original_moe.experts
        else:
            # Create new experts (for from_config initialization)
            from transformers.models.olmoe.modeling_olmoe import OlmoeMLP
            self.experts = nn.ModuleList([OlmoeMLP(config) for _ in range(self.original_num_experts)])
        
        # Create expanded gate
        self.gate = nn.Linear(config.hidden_size, self.total_num_experts, bias=False)
        
        # Initialize gate weights
        if original_moe is not None:
            # Copy original gate weights
            self.gate.weight.data[:self.original_num_experts, :] = (
                original_moe.gate.weight.data.to(self.gate.weight.dtype)
            )
        else:
            # Initialize original expert weights normally
            nn.init.normal_(
                self.gate.weight.data[:self.original_num_experts, :],
                mean=0.0,
                std=config.initializer_range
            )
        
        # Initialize trash can expert weights
        nn.init.normal_(
            self.gate.weight.data[self.original_num_experts:, :],
            mean=config.trash_can_init_mean,
            std=config.trash_can_init_std
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
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
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
    """Decoder layer with TrashCanSparseMoeBlock."""
    
    def __init__(self, config: SelectMoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        
        # Replace the MLP with TrashCanSparseMoeBlock
        self.mlp = TrashCanSparseMoeBlock(config)


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

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [SelectMoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # Import and use the same components as OlmoeModel
        from transformers.models.olmoe.modeling_olmoe import OlmoeRMSNorm, OlmoeRotaryEmbedding
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Simple implementation for cache and attention mask handling
        if cache_position is None:
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Create causal attention mask
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
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
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)
        
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


def custom_constraint_loss(router_logits: List[torch.Tensor], config: SelectMoeConfig) -> torch.Tensor:
    """Compute custom constraint loss for trash can experts."""
    if len(router_logits) == 0:
        return torch.tensor(0.0, device=next(iter(router_logits)).device if router_logits else 'cpu')
    
    total_loss = 0.0
    num_layers = 0
    
    for layer_router_logits in router_logits:
        if layer_router_logits is None:
            continue
            
        # Get routing probabilities
        routing_probs = F.softmax(layer_router_logits, dim=-1)
        
        # Compute top-k expert selection probabilities (excluding trash can experts)
        regular_expert_probs = routing_probs[:, :config.num_experts]
        top_k_probs, _ = torch.topk(regular_expert_probs, k=config.num_experts_per_tok, dim=-1)
        ratio = top_k_probs.sum(dim=-1)  # Shape: (batch_size * seq_len,)
        
        # Beta distribution inspired loss: -((alpha - 1) * log(ratio) + (beta - 1) * log(1 - ratio))
        # With alpha = 1, this simplifies to -(beta - 1) * log(1 - ratio)
        alpha = config.trash_can_loss_alpha
        beta = config.trash_can_loss_beta
        
        # Clamp ratio to avoid log(0)
        ratio_clamped = torch.clamp(ratio, min=1e-8, max=1.0 - 1e-8)
        
        if alpha == 1.0:
            # Simplified form when alpha = 1
            layer_loss = -(beta - 1) * torch.log(1 - ratio_clamped)
        else:
            # Full beta distribution form
            layer_loss = -((alpha - 1) * torch.log(ratio_clamped) + (beta - 1) * torch.log(1 - ratio_clamped))
        
        total_loss += layer_loss.mean()
        num_layers += 1
    
    return total_loss / max(num_layers, 1)


class SelectMoeForCausalLM(SelectMoePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = SelectMoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        # Note: total experts include trash can experts for load balancing
        self.num_experts = config.num_experts + config.num_experts_per_tok  # trash_can_experts_count
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
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
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

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            
            # Add custom constraint loss for trash can experts
            constraint_loss = custom_constraint_loss(
                outputs.router_logits if return_dict else outputs[-1],
                self.config
            )
            aux_loss += self.config.constraint_loss_weight * constraint_loss
            
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


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