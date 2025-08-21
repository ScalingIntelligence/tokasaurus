import torch
from torch import Tensor, nn
from transformers import Qwen3Config

from tokasaurus.model.llama import (
    LlamaAttention,
    LlamaBlock,
    LlamaForCausalLM,
    LlamaModel,
    apply_rotary_pos_emb,
    reduce_scatter,
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
        return self.weight * hidden_states.to(input_dtype)


class Qwen3Attention(LlamaAttention):
    qkv_bias: bool = False  # Qwen3 doesn't use bias in projection layers

    def __init__(self, config, layer_idx, extra_config):
        super().__init__(config, layer_idx, extra_config)

        # Add query and key normalization as in Qwen3
        self.q_norm = Qwen3RMSNorm(self.head_dim(), config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim(), config.rms_norm_eps)

    def head_dim(self):
        # Qwen3 uses explicit head_dim in the config, instead of inferring it from hidden_size and num_attention_heads
        return self.config.head_dim

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
        head_dim = self.head_dim()

        # Project to query, key, value
        query_proj = self.q_proj(hidden_states)
        key_proj = self.k_proj(hidden_states)
        value_proj = self.v_proj(hidden_states)

        # In tokasaurus, hidden_states is already 2D: [total_tokens, hidden_size]
        # We need to reshape to [total_tokens, num_heads, head_dim] for normalization
        # then back to [total_tokens, num_heads, head_dim] for attention

        # Match HuggingFace exactly: q_proj -> view -> q_norm
        # Note: In tokasaurus, bsz represents total tokens, not batch size
        # HF uses: query_proj.view(batch_size, seq_len, num_heads, head_dim)
        # We use: query_proj.view(total_tokens, 1, num_heads, head_dim) since each "token" is independent
        query_states = self.q_norm(
            query_proj.view(bsz, 1, self.num_attention_heads, head_dim)
        )
        key_states = self.k_norm(key_proj.view(bsz, 1, self.num_kv_heads, head_dim))

        # Flatten back to [total_tokens, num_heads, head_dim] for tokasaurus attention
        query_states = query_states.view(bsz, self.num_attention_heads, head_dim)
        key_states = key_states.view(bsz, self.num_kv_heads, head_dim)
        value_states = value_proj.view(bsz, self.num_kv_heads, head_dim)

        # Store original dtype
        dtype = query_states.dtype

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

        raw_attn_output = self.attn_fn(
            query_states,
            key_states,
            value_states,
            self.layer_cache.k_cache,
            self.layer_cache.v_cache,
        ).clone()

        attn_output = raw_attn_output.view(bsz, self.num_attention_heads * head_dim)

        # NOTE: The purpose of running prefill tokens through the model is only
        # to populate the kv cache. After this last layer, we don't need to
        # do any more compute with these tokens. Technically, we could have
        # skipped the sdpa call for these too, but that would screw with the
        # paging information.
        if (
            self.layer_idx == self.config.num_hidden_layers - 1
            and self.extra_config.tp_size == 1
        ):
            attn_output = attn_output[batch_state.lm_head_indices]
            residual = residual[batch_state.lm_head_indices]

        o_proj = self.o_proj(attn_output)

        o_proj = reduce_scatter(o_proj, self.extra_config)

        with_residual = residual + o_proj

        batch_state.hidden_states = with_residual
        return batch_state


class Qwen3Block(LlamaBlock):
    attn_cls = Qwen3Attention


class Qwen3Model(LlamaModel):
    block_cls = Qwen3Block


class Qwen3ForCausalLM(LlamaForCausalLM):
    model_cls = Qwen3Model
    config_cls = Qwen3Config

    def head_dim(self):
        return self.config.head_dim

    def make_name_to_hf_name(self):
        """Override to add q_norm and k_norm parameter mappings"""
        name_to_hf_name = super().make_name_to_hf_name()

        # Add mappings for q_norm and k_norm in each attention layer
        for layer_idx in range(self.config.num_hidden_layers):
            name_to_hf_name[f"model.layers.{layer_idx}.self_attn.q_norm.weight"] = (
                f"model.layers.{layer_idx}.self_attn.q_norm.weight"
            )
            name_to_hf_name[f"model.layers.{layer_idx}.self_attn.k_norm.weight"] = (
                f"model.layers.{layer_idx}.self_attn.k_norm.weight"
            )

        return name_to_hf_name
