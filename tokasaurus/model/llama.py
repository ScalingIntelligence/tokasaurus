from pathlib import Path

import huggingface_hub
import torch
import torch.nn.functional as F
import transformers
from accelerate import init_empty_weights
from flashinfer import (
    sampling_from_probs,
)
from torch import Tensor, nn
from torch.distributed import _functional_collectives as funcol
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from tokasaurus.model.attention_utils import create_wrappers, tokasaurus_attention
from tokasaurus.model.kv_cache import LayerKVCache
from tokasaurus.model.safetensors_utils import (
    can_load_from_safetensors,
    load_safetensors_repo,
)
from tokasaurus.model.types import (
    AttentionInfo,
    BatchState,
    DeviceType,
    ExtraModelConfig,
    ModelOutputTensors,
    WrapperCollection,
)


class RMSNorm(nn.Module):
    def __init__(self, config: LlamaConfig):
        """
        Taken from LlamaRMSNorm.
        """
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.ones(config.hidden_size))

    def forward(self, hidden_states: Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.config.rms_norm_eps)

        if self.weight is not None:
            return self.weight * hidden_states.to(input_dtype)
        else:
            return hidden_states.to(input_dtype)


def all_gather(x: Tensor, extra_config: ExtraModelConfig):
    if extra_config.tp_size == 1:
        return x

    assert extra_config.tp_group is not None

    # funcol + no compile + tp > 1 and pp > 1 leads to some nccl crash
    if extra_config.torch_compile:
        return funcol.all_gather_tensor(x, gather_dim=0, group=extra_config.tp_group)

    out = torch.empty(
        (extra_config.tp_size * x.shape[0], *x.shape[1:]),
        device=x.device,
        dtype=x.dtype,
    )
    torch.distributed.all_gather_into_tensor(out, x, group=extra_config.tp_group)
    return out


def reduce_scatter(x: Tensor, extra_config: ExtraModelConfig):
    if extra_config.tp_size == 1:
        return x

    assert extra_config.tp_group is not None

    # funcol + no compile + tp > 1 and pp > 1 leads to some nccl crash
    if extra_config.torch_compile:
        return funcol.reduce_scatter_tensor(
            x, reduceOp="sum", scatter_dim=0, group=extra_config.tp_group
        )

    out = torch.empty(
        (x.shape[0] // extra_config.tp_size, *x.shape[1:]),
        device=x.device,
        dtype=x.dtype,
    )
    torch.distributed.reduce_scatter_tensor(out, x, group=extra_config.tp_group)
    return out


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # this is the one difference between the llama
    # and the qwen2 architectures
    qkv_bias: bool = False

    def __init__(
        self, config: LlamaConfig, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config)

        self.tp_size = extra_config.tp_size or 1

        assert config.num_attention_heads % self.tp_size == 0

        assert self.config.num_attention_heads % self.tp_size == 0
        assert (
            self.config.num_key_value_heads % self.tp_size == 0
            or self.config.num_key_value_heads == 1
        )

        self.num_attention_heads = config.num_attention_heads // self.tp_size
        self.num_kv_heads = (
            config.num_key_value_heads // self.tp_size
            if config.num_key_value_heads > 1
            else 1
        )

        self.q_proj = nn.Linear(
            self.config.hidden_size,
            self.num_attention_heads * self.head_dim(),
            bias=self.qkv_bias,
        )
        self.k_proj = nn.Linear(
            self.config.hidden_size,
            self.num_kv_heads * self.head_dim(),
            bias=self.qkv_bias,
        )
        self.v_proj = nn.Linear(
            self.config.hidden_size,
            self.num_kv_heads * self.head_dim(),
            bias=self.qkv_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim(),
            config.hidden_size,
            bias=False,
        )

        self.layer_cache: LayerKVCache | None = None

        self.wrapper_collection: WrapperCollection | None = None
        self.attention_info: AttentionInfo | None = None

        self.attn_fn = self.make_attn_fn()

    def head_dim(self):
        return self.config.hidden_size // self.config.num_attention_heads

    def make_attn_fn(self):
        @torch.library.custom_op(
            f"tokasaurus::llama_attention_layer_{self.layer_idx}",
            mutates_args=("k_cache", "v_cache"),
        )
        def attn_fn(
            ragged_q: Tensor,
            ragged_k: Tensor,
            ragged_v: Tensor,
            k_cache: Tensor,
            v_cache: Tensor,
        ) -> Tensor:
            assert self.attention_info is not None
            assert self.wrapper_collection is not None

            attention_info = self.attention_info
            num_padding = attention_info.num_padding

            if torch.cuda.is_current_stream_capturing():
                assert num_padding == 0, f"no sirree, num_padding={num_padding}"

            orig_q = ragged_q
            if num_padding > 0:
                ragged_q = ragged_q[:-num_padding]
                ragged_k = ragged_k[:-num_padding]
                ragged_v = ragged_v[:-num_padding]

            out = tokasaurus_attention(
                ragged_q=ragged_q,
                ragged_k=ragged_k,
                ragged_v=ragged_v,
                k_cache=k_cache,
                v_cache=v_cache,
                attn_info=attention_info,
                wrappers=self.wrapper_collection,
            )

            if num_padding > 0:
                out = F.pad(out, (0, 0, 0, 0, 0, num_padding))

            assert out.shape == orig_q.shape, (
                f"out.shape={out.shape} != orig_q.shape={orig_q.shape}"
            )
            return out

        @attn_fn.register_fake
        def _(
            ragged_q: Tensor,
            ragged_k: Tensor,
            ragged_v: Tensor,
            k_cache: Tensor,
            v_cache: Tensor,
        ) -> Tensor:
            return torch.empty_like(ragged_q)

        return attn_fn

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

        hidden_states = all_gather(hidden_states, self.extra_config)
        bsz = hidden_states.shape[0]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, self.num_attention_heads, -1)
        key_states = key_states.view(bsz, self.num_kv_heads, -1)
        value_states = value_states.view(bsz, self.num_kv_heads, -1)

        cos, sin = batch_state.position_embeddings

        dtype = query_states.dtype

        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            unsqueeze_dim=1,  # unsqueeze dim = head dim on q/k
        )

        query_states = query_states.to(dtype)
        key_states = key_states.to(dtype)

        raw_attn_output = self.attn_fn(
            ragged_q=query_states,
            ragged_k=key_states,
            ragged_v=value_states,
            k_cache=self.layer_cache.k_cache,
            v_cache=self.layer_cache.v_cache,
        ).clone()  # HACK: torch compile crashes with some weakref RuntimeError without this clone.

        attn_output = raw_attn_output.view(bsz, -1)

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


class LlamaMLP(nn.Module):
    def __init__(
        self, config: LlamaConfig, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.tp_size = extra_config.tp_size
        assert self.config.intermediate_size % self.tp_size == 0
        self.intermediate_size = self.config.intermediate_size // self.tp_size

        self.up_proj = nn.Linear(
            self.config.hidden_size,
            self.intermediate_size,
            bias=False,
        )
        self.gate_proj = nn.Linear(
            config.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size,
            config.hidden_size,
            bias=False,
        )

        self.input_layernorm = RMSNorm(config)

    def forward(
        self,
        batch_state: BatchState,
    ):
        inp = batch_state.hidden_states
        assert inp is not None
        hidden_states = self.input_layernorm(inp)

        hidden_states = all_gather(hidden_states, self.extra_config)

        up = self.up_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        prod = F.silu(gate) * up
        down = self.down_proj(prod)

        down = reduce_scatter(down, self.extra_config)

        with_residual = inp + down

        batch_state.hidden_states = with_residual
        return batch_state


class LlamaBlock(nn.Module):
    attn_cls = LlamaAttention

    def __init__(
        self, config: LlamaConfig, extra_config: ExtraModelConfig, layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.layer_idx = layer_idx

        self.self_attn = self.attn_cls(config, extra_config, layer_idx)
        self.mlp = LlamaMLP(config, extra_config, layer_idx)

    def forward(self, batch_state: BatchState):
        out = self.self_attn(batch_state)
        out = self.mlp(out)
        return out


@torch.library.custom_op("tokasaurus::sample_from_probs", mutates_args=())
def sample_from_probs(probs: Tensor) -> Tensor:
    batch_size = probs.shape[0]
    uniform = torch.rand(batch_size, device=probs.device, dtype=probs.dtype)
    samples = sampling_from_probs(probs, uniform)
    return samples


@sample_from_probs.register_fake
def _(probs: Tensor) -> Tensor:
    batch_size = probs.shape[0]
    out = torch.empty(batch_size, device=probs.device, dtype=torch.int32)
    return out


def calc_tokens_and_logprobs(
    logits: Tensor,
    temperature: Tensor,
    greedy_mask: Tensor,
    config: ExtraModelConfig,
):
    augmented_logits = logits

    if temperature is not None:
        augmented_logits = augmented_logits / temperature.unsqueeze(-1)

    probs = F.softmax(augmented_logits, dim=-1)
    next_token_ids = sample_from_probs(probs).long()

    greedy_ids = logits.argmax(dim=-1)

    next_token_ids = torch.where(
        greedy_mask,
        greedy_ids,
        next_token_ids,
    )

    if config.enable_chosen_logprobs:
        # TODO: because this is all in fp32, I think the numerics are ok here.
        chosen_probs = probs.gather(dim=-1, index=next_token_ids.unsqueeze(-1)).squeeze(
            -1
        )
        chosen_logprobs = chosen_probs.log()
    else:
        chosen_logprobs = None

    topk = config.topk_logprobs
    if topk is not None:
        assert topk > 0
        topk_probs, topk_indices = torch.topk(probs, k=topk, dim=-1)
        topk_tokens = topk_indices
        topk_logprobs = topk_probs.log()
    else:
        topk_tokens = None
        topk_logprobs = None

    return ModelOutputTensors(
        output_ids=next_token_ids,
        chosen_logprobs=chosen_logprobs,
        topk_indices=topk_tokens,
        topk_logprobs=topk_logprobs,
    )


class LlamaLMHead(nn.Module):
    def __init__(self, config: LlamaConfig, extra_config: ExtraModelConfig):
        super().__init__()
        self.config = config
        self.extra_config = extra_config

        self.input_norm = RMSNorm(config)

        self.tp_size = extra_config.tp_size or 1

        assert config.vocab_size % self.tp_size == 0
        head_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, head_size, bias=False)

    def forward(self, batch_state: BatchState):
        assert batch_state.hidden_states is not None
        assert batch_state.lm_head_indices is not None

        hidden_states = batch_state.hidden_states

        if batch_state.lm_head_indices.numel() == 0:
            next_token_ids = torch.empty(
                0, device=hidden_states.device, dtype=torch.long
            )
            chosen_logprobs = torch.empty(
                0, device=hidden_states.device, dtype=torch.float32
            )
            outputs = ModelOutputTensors(
                output_ids=next_token_ids,
                chosen_logprobs=chosen_logprobs,
            )
        else:
            if self.extra_config.tp_size > 1:
                hidden_states = all_gather(hidden_states, self.extra_config)
                needed_hidden_states = hidden_states[batch_state.lm_head_indices]
            else:
                needed_hidden_states = hidden_states

            hidden_states = self.input_norm(needed_hidden_states)

            logits = self.lm_head(hidden_states).float()
            # if not torch.cuda.is_current_stream_capturing():
            #     print("batch_state:", batch_state)
            #     if 32767 in batch_state.attention_info.prefill_info.kv_indices:
            #         print("llama model logits:", logits)
            #         torch.save(logits, "/tmp/tokasaurus_logits.pt")
            #         raise Exception("Stop here")

            assert batch_state.sampling_params.top_p is None
            assert batch_state.sampling_params.temperature is not None
            assert batch_state.sampling_params.greedy_mask is not None

            outputs = calc_tokens_and_logprobs(
                logits,
                temperature=batch_state.sampling_params.temperature,
                greedy_mask=batch_state.sampling_params.greedy_mask,
                config=self.extra_config,
            )

            # TODO: fuse these small all_gathers into a single op
            # TODO: with the exception of output_ids, we don't need to all-gather
            # the other tensors, we can simply gather them to tp rank 0.
            outputs.output_ids = all_gather(outputs.output_ids, self.extra_config)

            if outputs.chosen_logprobs is not None:
                outputs.chosen_logprobs = all_gather(
                    outputs.chosen_logprobs, self.extra_config
                )

            if outputs.topk_indices is not None:
                outputs.topk_indices = all_gather(
                    outputs.topk_indices, self.extra_config
                )

            if outputs.topk_logprobs is not None:
                outputs.topk_logprobs = all_gather(
                    outputs.topk_logprobs, self.extra_config
                )

        # TODO does "last layer hidden states" typically refer to
        # the activations after the lm head's norm? if so we should update
        # the state (that being said, we don't return hidden states for now).
        batch_state.outputs = outputs
        return batch_state


class LlamaEmbeddings(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, batch_state: BatchState):
        hidden_states = self.embed_tokens(batch_state.input_ids)

        batch_state.hidden_states = hidden_states
        return batch_state


class LlamaModel(nn.Module):
    block_cls = LlamaBlock
    rope_cos: Tensor
    rope_sin: Tensor

    def __init__(
        self,
        config: LlamaConfig,
        extra_config: ExtraModelConfig,
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config

        layer_start, layer_end = calc_pipeline_layer_start_and_end(
            config,
            extra_config,
        )

        if layer_start == 0:
            self.embed_tokens = LlamaEmbeddings(config)
        else:
            self.embed_tokens = nn.Identity()

        if layer_end is None:
            layer_end = config.num_hidden_layers

        print(f"Building layers {layer_start} to {layer_end}")

        layers = []
        for i in range(config.num_hidden_layers):
            if i >= layer_start and i < layer_end:
                layers.append(self.block_cls(config, extra_config, i))
            else:
                layers.append(nn.Identity())

        self.layers = nn.ModuleList(layers)

        self.rope = LlamaRotaryEmbedding(
            config=config,
        )

        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        dummy_float_input = torch.empty((0, config.hidden_size), dtype=torch.float32)

        cos, sin = self.rope(dummy_float_input, position_ids)
        assert cos.dtype == torch.float32
        assert sin.dtype == torch.float32
        self.register_buffer("rope_cos", cos.squeeze(0), persistent=False)
        self.register_buffer("rope_sin", sin.squeeze(0), persistent=False)

    def forward(self, batch_state: BatchState):
        out: BatchState = self.embed_tokens(batch_state)
        assert self.rope_cos.dtype == torch.float32
        assert self.rope_sin.dtype == torch.float32
        cos = self.rope_cos[batch_state.position_ids]
        sin = self.rope_sin[batch_state.position_ids]
        out.position_embeddings = (cos, sin)

        for layer_idx, layer in enumerate(self.layers):
            out = layer(out)
            if not torch.cuda.is_current_stream_capturing() and layer_idx == 0:
                torch.save(out.hidden_states, "/tmp/tokasaurus_layer_0_hidden_states.pt")
        return out


def calc_pipeline_layer_start_and_end(
    config: LlamaConfig,
    extra_config: ExtraModelConfig,
):
    if extra_config.pp_size > 1:
        assert extra_config.pp_rank is not None

        avg_layers_per_stage = config.num_hidden_layers / extra_config.pp_size
        layer_start = round(extra_config.pp_rank * avg_layers_per_stage)
        layer_end = round((extra_config.pp_rank + 1) * avg_layers_per_stage)

    else:
        layer_start = 0
        layer_end = None

    return layer_start, layer_end


class LlamaForCausalLM(nn.Module):
    model_cls = LlamaModel
    config_cls = LlamaConfig

    def __init__(
        self,
        config: LlamaConfig,
        extra_config: ExtraModelConfig,
    ):
        super().__init__()
        self.config = config
        self.extra_config = extra_config
        self.device = torch.get_default_device()
        self.dtype = torch.get_default_dtype()

        self.model = self.model_cls(config, extra_config)

        _, layer_end = calc_pipeline_layer_start_and_end(
            config,
            extra_config,
        )
        if layer_end in [None, config.num_hidden_layers]:
            self.lm_head = LlamaLMHead(config, extra_config)
        else:
            self.lm_head = nn.Identity()

    def num_kv_heads(self):
        all_heads = self.config.num_key_value_heads
        tp_size = self.extra_config.tp_size
        assert all_heads % tp_size == 0
        return all_heads // tp_size

    def num_qo_heads(self):
        all_heads = self.config.num_attention_heads
        tp_size = self.extra_config.tp_size
        assert all_heads % tp_size == 0
        return all_heads // tp_size

    def head_dim(self):
        return self.config.hidden_size // self.config.num_attention_heads

    def forward(
        self,
        batch_state: BatchState,
        async_tp: bool = False,
    ):
        self.async_tp = async_tp

        # making a copy of the input state - needed when combining cudagraphs + pp,
        # where we need to keep track of references to both the input
        # and output hidden states.

        out = BatchState(
            input_ids=batch_state.input_ids,
            attention_info=batch_state.attention_info,
            position_ids=batch_state.position_ids,
            hidden_states=batch_state.hidden_states,
            lm_head_indices=batch_state.lm_head_indices,
            sampling_params=batch_state.sampling_params,
        )

        out = self.model(out)
        out = self.lm_head(out)

        return out

    def set_wrappers(self, wrappers: WrapperCollection):
        self.wrapper_collection = wrappers
        for layer in self.model.modules():
            if isinstance(layer, LlamaAttention):
                layer.wrapper_collection = wrappers

    def setup_caches(self, num_pages: int, page_size: int):
        wrappers = create_wrappers(
            device=self.device,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
        )
        self.set_wrappers(wrappers)
        for layer in self.model.modules():
            if isinstance(layer, LlamaAttention):
                layer.layer_cache = LayerKVCache(
                    head_dim=layer.head_dim(),
                    num_kv_heads=layer.num_kv_heads,
                    num_pages=num_pages,
                    page_size=page_size,
                    device=self.device,
                    dtype=self.dtype,
                )

    def set_attention_info(self, attn_info: AttentionInfo):
        for layer in self.model.modules():
            if isinstance(layer, LlamaAttention):
                layer.attention_info = attn_info

    def plan(self, attn_info: AttentionInfo, non_blocking: bool = False):
        wrappers = self.wrapper_collection
        assert wrappers is not None

        self.set_attention_info(attn_info)

        head_dim = self.head_dim()
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

    def to(self, device: DeviceType | None = None, dtype: torch.dtype | None = None):  # type: ignore
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return super().to(device=device, dtype=dtype)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        extra_config: ExtraModelConfig | None = None,
        device: DeviceType | None = None,
        dtype: torch.dtype | None = None,
    ):
        if extra_config is None:
            extra_config = ExtraModelConfig()

        config: LlamaConfig = cls.config_cls.from_pretrained(model_name_or_path)  # type: ignore
        if extra_config.rope_scaling is not None:
            config.rope_scaling = extra_config.rope_scaling

        with init_empty_weights(include_buffers=False):
            model = cls(
                config,
                extra_config,
            )
        model.dtype = dtype
        model.device = device

        if (as_path := Path(model_name_or_path)).exists():
            model_path = as_path
        else:
            snapshot_path_str = huggingface_hub.snapshot_download(
                model_name_or_path,
                allow_patterns=["*.safetensors", "*.json"],
            )

            model_path = Path(snapshot_path_str)

        if can_load_from_safetensors(model_path):
            print("Loading from safetensors")
            model.load_from_safetensors(model_path)
        else:
            print("Loading from hf")
            model.load_from_hf_pretrained(model_path, device, dtype)

        # SE (10/18/24): It is important not to call model.to(device, dtype) because
        # this will convert the `inv_freq` buffer in the rotary embeddings to fp16
        # the HF load from pretrained is careful to not do this and keeps it in fp32.
        # The dtype for the parameters is already handled by the load calls above, but
        # it's possible that there are other buffers which *should* be converted to fp16.
        # TODO: it's probably easiest to figure out how we can just use HFs `load_from_pretrained`
        # to load the model weights so we can ensure that there are no other subtle differences
        model.to(device=device)

        return model

    def make_name_to_hf_name(self):
        keys = self.state_dict().keys()

        name_to_hf_name = {k: k for k in keys}

        for layer_idx in range(self.config.num_hidden_layers):
            name_to_hf_name[
                f"model.layers.{layer_idx}.self_attn.input_layernorm.weight"
            ] = f"model.layers.{layer_idx}.input_layernorm.weight"
            name_to_hf_name[f"model.layers.{layer_idx}.mlp.input_layernorm.weight"] = (
                f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            )

        name_to_hf_name["model.embed_tokens.embed_tokens.weight"] = (
            "model.embed_tokens.weight"
        )
        name_to_hf_name["lm_head.input_norm.weight"] = "model.norm.weight"

        if self.config.tie_word_embeddings:
            name_to_hf_name["lm_head.lm_head.weight"] = "model.embed_tokens.weight"
        else:
            name_to_hf_name["lm_head.lm_head.weight"] = "lm_head.weight"

        return name_to_hf_name

    def tp_modify_state_dict(self, state_dict: dict, tp_rank: int, tp_size: int):
        tp_map = self.make_tp_map()

        for param_name, param in state_dict.items():
            if (split_dim := tp_map.get(param_name)) is not None:
                state_dict[param_name] = param.chunk(tp_size, dim=split_dim)[tp_rank]

        return state_dict

    def make_tp_map(self):
        """
        Maps parameter names to the dimension they should be split on.
        Parameters that are not included in the map should not be split.
        """

        tp_map = {}
        for param_name, _ in self.named_parameters():
            if any(
                param_name.endswith(suffix)
                for suffix in [
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "up_proj.weight",
                    "gate_proj.weight",
                ]
            ):
                tp_map[param_name] = 0

            elif any(
                param_name.endswith(suffix)
                for suffix in ["o_proj.weight", "down_proj.weight"]
            ):
                tp_map[param_name] = 1

        return tp_map

    def load_from_hf_pretrained(
        self,
        model_name_or_path: str | Path,
        device: DeviceType | None = None,
        dtype: torch.dtype | None = None,
    ):
        hf_model: transformers.LlamaForCausalLM = (
            transformers.AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                device_map=device,
            )
        )

        state_dict = hf_model.state_dict()
        patched_state_dict = state_dict.copy()

        name_to_hf_name = self.make_name_to_hf_name()

        for name, hf_name in name_to_hf_name.items():
            if name != hf_name:
                patched_state_dict[name] = state_dict[hf_name]
                patched_state_dict.pop(hf_name)

        tp_size = self.extra_config.tp_size
        tp_rank = self.extra_config.tp_rank

        if tp_size > 1:
            assert tp_rank is not None
            patched_state_dict = self.tp_modify_state_dict(
                patched_state_dict, tp_rank, tp_size
            )

        # must assign because of meta parameters
        self.load_state_dict(patched_state_dict, assign=True, strict=False)

    def load_from_safetensors(
        self,
        model_path: Path,
    ):
        print("Loading from safetensors model_path", model_path)
        name_to_hf_name = self.make_name_to_hf_name()
        all_hf_names = set(name_to_hf_name.values())

        hf_state_dict = load_safetensors_repo(
            model_path,
            include_parameters=all_hf_names,
            device=self.device,
            tp_rank=self.extra_config.tp_rank,
            tp_size=self.extra_config.tp_size,
            tp_map=self.make_tp_map(),
        )

        state_dict = {k: hf_state_dict[v] for k, v in name_to_hf_name.items()}

        self.load_state_dict(state_dict, assign=True, strict=False)
