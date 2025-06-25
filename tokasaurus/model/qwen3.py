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
from tokasaurus.model.types import AttentionInfo, BatchState


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
        super().__init__(config, layer_idx, extra_config)
        
        # Override head_dim if explicitly set in config (Qwen3 uses explicit head_dim)
        if hasattr(config, 'head_dim') and config.head_dim is not None:
            self.head_dim = config.head_dim
            
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

        # Project to query, key, value
        query_proj = self.q_proj(hidden_states)
        key_proj = self.k_proj(hidden_states)
        value_proj = self.v_proj(hidden_states)

        # Reshape and apply normalization (following HF implementation order)
        query_states = query_proj.view(bsz, self.num_attention_heads, self.head_dim)
        key_states = key_proj.view(bsz, self.num_kv_heads, self.head_dim)
        value_states = value_proj.view(bsz, self.num_kv_heads, self.head_dim)

        # Store original dtype before normalization
        dtype = query_states.dtype

        # Apply query and key normalization before rotary embeddings (as in HF)
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

        raw_attn_output = self.attn_fn(
            query_states,
            key_states,
            value_states,
            self.layer_cache.k_cache,
            self.layer_cache.v_cache,
        ).clone()

        attn_output = raw_attn_output.view(bsz, self.num_attention_heads * self.head_dim)

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

        from tokasaurus.model.llama import reduce_scatter
        attn_output = reduce_scatter(attn_output, self.extra_config)

        output = self.o_proj(attn_output)
        output = output.to(dtype)

        with_residual = residual + output

        batch_state.hidden_states = with_residual
        return batch_state


class Qwen3Block(LlamaBlock):
    attn_cls = Qwen3Attention


class Qwen3Model(LlamaModel):
    block_cls = Qwen3Block


class Qwen3ForCausalLM(LlamaForCausalLM):
    model_cls = Qwen3Model
    config_cls = Qwen3Config

    def plan(self, attn_info, non_blocking: bool = False):
        """Override to use explicit head_dim from config instead of calculated value"""
        wrappers = self.wrapper_collection
        assert wrappers is not None

        for layer in self.model.modules():
            if isinstance(layer, LlamaAttention):
                layer.attention_info = attn_info

        # Use explicit head_dim from config if available (Qwen3 specific)
        if hasattr(self.config, 'head_dim') and self.config.head_dim is not None:
            head_dim = self.config.head_dim
        else:
            head_dim = self.config.hidden_size // self.config.num_attention_heads

        num_qo_heads = self.num_qo_heads()
        num_kv_heads = self.num_kv_heads()

        page_size = attn_info.page_size
        q_data_type = self.dtype
        kv_data_type = self.dtype

        if (
            prefill_info := attn_info.prefill_info
        ) is not None and prefill_info.num_tokens > 0:
            assert prefill_info.qo_indptr is not None
            wrappers.prefill_wrapper.plan(
                qo_indptr=prefill_info.qo_indptr,
                paged_kv_indptr=prefill_info.kv_indptr,
                paged_kv_indices=prefill_info.kv_indices,
                paged_kv_last_page_len=prefill_info.kv_last_page_len,
                num_kv_heads=num_kv_heads,
                num_qo_heads=num_qo_heads,
                head_dim=head_dim,
                page_size=page_size,
                q_data_type=q_data_type,
                kv_data_type=kv_data_type,
                causal=True,
                non_blocking=non_blocking,
            )

        if (
            hydragen_info := attn_info.hydragen_info
        ) is not None and hydragen_info.num_tokens > 0:
            assert hydragen_info.qo_indptr is not None
            wrappers.hydragen_wrapper.plan(
                qo_indptr=hydragen_info.qo_indptr,
                paged_kv_indptr=hydragen_info.kv_indptr,
                paged_kv_indices=hydragen_info.kv_indices,
                paged_kv_last_page_len=hydragen_info.kv_last_page_len,
                num_kv_heads=num_kv_heads,
                num_qo_heads=num_qo_heads,
                head_dim=head_dim,
                page_size=page_size,
                q_data_type=q_data_type,
                kv_data_type=kv_data_type,
                causal=False,
                non_blocking=non_blocking,
            )

        if (
            decode_info := attn_info.decode_info
        ) is not None and decode_info.num_tokens > 0:
            wrappers.decode_wrapper.plan(
                indptr=decode_info.kv_indptr,
                indices=decode_info.kv_indices,
                last_page_len=decode_info.kv_last_page_len,
                num_kv_heads=num_kv_heads,
                num_qo_heads=num_qo_heads,
                head_dim=head_dim,
                page_size=page_size,
                q_data_type=q_data_type,
                kv_data_type=kv_data_type,
                non_blocking=non_blocking,
            )

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