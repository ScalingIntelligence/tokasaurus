try:
    from transformers import Qwen3Config
except ImportError:
    # Fallback for older transformers versions
    from transformers import Qwen2Config as Qwen3Config

import torch
from torch import Tensor, nn

from tokasaurus.model.llama import (
    LlamaAttention,
    LlamaBlock,
    LlamaForCausalLM,
    LlamaModel,
    apply_rotary_pos_emb,
)
from tokasaurus.model.types import BatchState


class Qwen3RMSNorm(nn.Module):
    """RMSNorm for head dimension (used in q_norm and k_norm)"""
    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, hidden_states: Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        output = self.weight * hidden_states
        return output.to(input_dtype)


class Qwen3Attention(LlamaAttention):
    qkv_bias: bool = False  # Qwen3 doesn't use bias in projection layers

    def __init__(self, config, layer_idx, extra_config):
        # Override to use explicit head_dim from config if available
        self.original_head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        
        # Temporarily remove head_dim from config so LlamaAttention doesn't use it incorrectly
        temp_head_dim = getattr(config, 'head_dim', None)
        if hasattr(config, 'head_dim'):
            delattr(config, 'head_dim')
        
        super().__init__(config, layer_idx, extra_config)
        
        # Restore head_dim to config
        if temp_head_dim is not None:
            config.head_dim = temp_head_dim
        
        # Override head_dim and recreate projection layers with correct dimensions
        self.head_dim = self.original_head_dim
        
        # Recreate projection layers with correct dimensions
        self.q_proj = nn.Linear(
            self.config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=self.qkv_bias,
        )
        self.k_proj = nn.Linear(
            self.config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=self.qkv_bias,
        )
        self.v_proj = nn.Linear(
            self.config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=self.qkv_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        
        # Add query and key normalization as in Qwen3
        self.q_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        batch_state: BatchState,
    ):
        assert batch_state.hidden_states is not None
        assert batch_state.position_embeddings is not None
        assert self.layer_cache is not None
        assert self.layer_cache.v_cache is not None

        inp = batch_state.hidden_states
        residual = inp

        hidden_states = self.input_layernorm(inp)

        from tokasaurus.model.llama import all_gather
        hidden_states = all_gather(hidden_states, self.extra_config)
        bsz = hidden_states.shape[0]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, self.num_attention_heads, -1)
        key_states = key_states.view(bsz, self.num_kv_heads, -1)
        value_states = value_states.view(bsz, self.num_kv_heads, -1)

        # Store original dtype before normalization
        dtype = query_states.dtype

        # Apply query and key normalization before rotary embeddings
        # Ensure dtype is preserved
        query_states = self.q_norm(query_states).to(dtype)
        key_states = self.k_norm(key_states).to(dtype)

        cos, sin = batch_state.position_embeddings

        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )

        # Ensure dtype is preserved after rotary embedding
        query_states = query_states.to(dtype)
        key_states = key_states.to(dtype)

        output = self.attn_fn(
            query_states,
            key_states,
            value_states,
            self.layer_cache.k_cache,
            self.layer_cache.v_cache,
        )

        from tokasaurus.model.llama import reduce_scatter
        output = reduce_scatter(output, self.extra_config)

        output = self.o_proj(output)
        output = output.to(dtype)

        output = residual + output

        return output


class Qwen3Block(LlamaBlock):
    attn_cls = Qwen3Attention


class Qwen3Model(LlamaModel):
    block_cls = Qwen3Block


class Qwen3ForCausalLM(LlamaForCausalLM):
    model_cls = Qwen3Model
    config_cls = Qwen3Config

    def make_name_to_hf_name(self):
        """Override to add q_norm and k_norm parameter mappings"""
        name_to_hf_name = super().make_name_to_hf_name()
        
        # Add mappings for q_norm and k_norm in each attention layer
        for layer_idx in range(self.config.num_hidden_layers):
            name_to_hf_name[f"model.layers.{layer_idx}.self_attn.q_norm.weight"] = \
                f"model.layers.{layer_idx}.self_attn.q_norm.weight"
            name_to_hf_name[f"model.layers.{layer_idx}.self_attn.k_norm.weight"] = \
                f"model.layers.{layer_idx}.self_attn.k_norm.weight"
        
        return name_to_hf_name