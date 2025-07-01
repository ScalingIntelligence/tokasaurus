from transformers import Qwen2Config

from tokasaurus.model.llama import (
    LlamaAttention,
    LlamaBlock,
    LlamaForCausalLM,
    LlamaModel,
)


class Qwen2Attention(LlamaAttention):
    qkv_bias: bool = True


class Qwen2Block(LlamaBlock):
    attn_cls = Qwen2Attention


class Qwen2Model(LlamaModel):
    block_cls = Qwen2Block


class Qwen2ForCausalLM(LlamaForCausalLM):
    model_cls = Qwen2Model
    config_cls = Qwen2Config

    def make_tp_map(self):
        """
        Need to add the qkv biases to the tp map.
        """
        tp_map = super().make_tp_map()
        for param_name, _ in self.named_parameters():
            if any(
                param_name.endswith(suffix)
                for suffix in [
                    "q_proj.bias",
                    "k_proj.bias",
                    "v_proj.bias",
                ]
            ):
                tp_map[param_name] = 0

        return tp_map
