from transformers import Qwen3Config

from tokasaurus.model.llama import (
    LlamaAttention,
    LlamaBlock,
    LlamaForCausalLM,
    LlamaModel,
)


class Qwen3Attention(LlamaAttention):
    qkv_bias: bool = True


class Qwen3Block(LlamaBlock):
    attn_cls = Qwen3Attention


class Qwen3Model(LlamaModel):
    block_cls = Qwen3Block


class Qwen3ForCausalLM(LlamaForCausalLM):
    model_cls = Qwen3Model
    config_cls = Qwen3Config
