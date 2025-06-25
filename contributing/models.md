# Adding a new model
As a running example, we'll describe adding support for the **Qwen-3** family of models.  
The same high-level steps apply to any model that can be framed as a (possibly light)
variant of the Llama / Qwen-2 architecture.

**Note**: This document is as much for human developers as it is for coding agents :) 
---

## 1. Implement the modelling file

All model specific code lives under `tokasaurus/model`.  For Qwen-3 we create
`tokasaurus/model/qwen3.py`. 

You should use a [reference implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modular_qwen3.py) from the `transformers` library as your guide.

Critical things to remember:
* The model must be compatible with the Tokasaurus `BatchState` interface.
* The model must use `tokasaurus_attention`, which calls to FlashInfer under the hood. 

---

## 2. Register the model type

Tokasaurus discovers models through a small mapping in
`tokasaurus/model/utils.py`.  Add an import and extend the dictionaries:

```diff
 # utils.py (near the top)
-from tokasaurus.model.qwen import Qwen2ForCausalLM
+from tokasaurus.model.qwen import Qwen2ForCausalLM
+from tokasaurus.model.qwen3 import Qwen3ForCausalLM
@@
-model_type = LlamaForCausalLM | Qwen2ForCausalLM
+model_type = LlamaForCausalLM | Qwen2ForCausalLM | Qwen3ForCausalLM
@@
     "qwen2": Qwen2ForCausalLM,
+    "qwen3": Qwen3ForCausalLM,
```

The key (`"qwen3"`) must match the `model_type` field inside the Hugging Face
`Qwen3Config` (you can verify via `AutoConfig.from_pretrained`).

---

## 3. (Optional) Extra features

If the new architecture requires deeper changes (e.g. different position
encoding or weight layout):

* Add new subclasses for the relevant modules (MLP, embeddings, …) similar to
  the Attention example above.
* Overwrite `make_name_to_hf_name` and/or `tp_modify_state_dict` in
  `LlamaForCausalLM` if the checkpoint key names differ.
* Add any device-side kernels you need in `tokasaurus/model/attention_utils.py`.

For purely additive features (e.g. support for **rope-scaling** parameters) you
usually only need to read the attribute from the HF `Config` and forward it to
`ExtraModelConfig`.

---

## 4. Tests

You have succeeded when the following command passes **without GPU OOMs or
assertion failures**:

```bash
MODEL=Qwen/Qwen3-0.6B pytest tests/test_logprobs.py -k test_logprobs -s
```

Tips to debug failures:
* Use the `--capture=no -s` flags for *verbose* test output.
* Run the server directly via `python -m tokasaurus.entry --config …` and send
a manual request with the OpenAI client.

AGENT
