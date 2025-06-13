# Tokasaurus: The Little (LLM) Engine That Could!


## Table of Contents

- [What is This?](#what-is-this)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Walkthrough of CLI Flags](#walkthrough-of-cli-flags)
- [System Design](#system-design)


## What is This?
Tokasaurus is an LLM inference engine designed for high-throughput workloads. Features include:

- OpenAI chat, completions, and batch APIs.
- Data, pipeline, and tensor parallelism (with support for [AsyncTP](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487)).
- Support for Llama3 and Qwen2 architectures.
- [Paged KV caching](https://arxiv.org/abs/2309.06180) with [prefix caching](https://arxiv.org/abs/2312.07104).
- Efficient attention over shared prefixes with [Hydragen](https://arxiv.org/abs/2402.05099), with automatic detection of shared prefixes across groups of sequences.
- End-to-end torch compile with dynamic shapes.
- CUDA graphs.
- Very low CPU overhead (important for small models/fast GPUs).
- A scheduler that can simulate the number of available KV cache blocks thousands of steps in the future, allowing us to aggressively onboard new sequences and keep our batch size as large as possible.
- No OOMs or recompiles in production: on engine startup, we launch a series of warmup inputs that trigger all torch recompiles ahead-of-time (torch will recompile whenever a tensor has an input dimension is 0 or 1) and make check for OOMs using the largest configured batch size.

NOTE: as a new project, expect some potentially rough edges :).

## Installation

Tokasaurus has been tested on Python >= 3.10. To install from PyPI, run:

```bash

pip install tokasaurus

```

Alternatively, clone the repo and run:

```bash

pip install -e .

```

## Quickstart

Once installed, you can launch the engine with:

```bash

# launch engine for Llama 1B (by default on port 10210).
toka model=meta-llama/Llama-3.2-1B-Instruct

# make a request to the engine (this command just wraps the OpenAI client)
toka-ping prompt='tell me a joke' max_tokens=256 chat=True

# launch a 70B model with pipeline parallelism across 8 gpus
toka model=meta-llama/Llama-3.1-70B-Instruct kv_cache_num_tokens='(512 * 1024)' pp_size=8
```

To ping the engine once it's been launched, you can use the OpenAI client:

```python

from openai import OpenAI
client = OpenAI(
    api_key='fake-key',
    base_url="http://0.0.0.0:10210/v1"
)
response = client.completions.create(
  model="default",
  prompt="On a dark desert highway, cool wind in my hair, warm smell of colitas, rising up through the air, up ahead in the distance, I saw a shimmering light, my head grew heavy and my sight grew dim, I had to stop for the night",
  temperature=0,
  n=2,
  max_tokens=100,
)

```

### LM Eval Harness

Since the engine supports the OpenAI API, you can plug it into the EleutherAI LM Eval harness using their local completions feature. First spin up an engine (see above) and then run evals on it with:

```bash

lm_eval --model local-completions --tasks gsm8k --model_args model=MODEL,base_url=http://0.0.0.0:PORT/v1/completions,num_concurrent=256,max_retries=3,tokenized_requests=False

```

## Walkthrough of CLI Flags

The tokasaurus CLI uses [Pydra](https://github.com/jordan-benjamin/pydra), which uses a `key=value` format to set config flags. It also allows for boolean shorthands (e.g. `key=T` is equivalent to `key=True`) and allows for Python expression evaluation between parentheses (e.g. `key='(2 * 1024)'` is equivalent to `key=2048`).

### The Basics

The only required parameter to launch an engine is the `model` field, which can point to a repo on HF or a local directory where a model is stored in HF format (just like when calling `from_pretrained` on a HF model). By default, the tokenizer will also be loaded using using the `model` flag. This can be overridden by setting the `tokenizer` flag yourself:

```bash
toka model=meta-llama/Llama-3.2-1B-Instruct

# e.g. if you want to load a fine-tuned model you saved to disk
toka model=my_local_dir tokenizer=meta-llama/Llama-3.2-1B-Instruct
```

### Leveraging Multiple GPUs

By default, the engine will only use a single GPU to serve the model. You can change this with the `dp_size`, `pp_size`, and `tp_size` flags to control data, pipeline, and tensor parallelism, respectively. These flags are composable: for example, `dp_size=2` and `pp_size=4` will use 8 GPUs in total by creating two data-parallel replicas that each contain 4 GPUs in a pipeline:

### Managing GPU Memory with KV Cache Limits and Concurrency Controls

The total amount of GPU memory used by the engine is the sum of GPU memory used to store the model weights, the activations, and the KV cache. While the model's GPU memory is fixed for a given model, we can control the size of the KV cache and the amount of activation memory we use.

The KV cache size is controlled with `kv_cache_size_num_tokens`, and we can cap activation memory with the flags `max_tokens_per_forward` and `max_seqs_per_forward`. With `max_tokens_per_forward`, you directly control the number of tokens being sent through the model in a single forward pass, which can include tokens from sequences running either prefill or decode. With `max_seqs_per_forward`, we control the total number of sequences that can be running (i.e. that are in prefill or in decode) at a given time. Importantly, this limits the number of tokens per forward pass that can ever be sent through the language modeling head of the model, which can have a disproportionately large impact on activation memory. Prefill tokens don't run through the LM head (since we don't need to decode anything from them), so they take less activation memory.

How should you tune these flags? Well, one of the most important factors for achieving high throughput is making the batch size as large as possible. A common bottleneck that limits the batch size in practice is the size of the KV cache - once your KV cache is full, you can't run any more sequences concurrently. Therefore, we want to make the KV cache as large as possible. However, in order to benefit from a large KV cache that can fit many sequences, we also must increase `max_seqs_per_forward` and `max_tokens_per_forward`. However, increasing these concurrency control flags increases the amount of used activation memory... decreasing the size of the largest KV cache we can fit.

In practice, what this means is that you should increase your KV cache size and concurrency control flags jointly, making sure that you're not excessively raising one without the other.

Note: when using multiple GPUs, these flags apply to each data-parallel replica separately (and apply collectively to all of the GPUs within a data parallel replica). For example, if you run with` dp_size=2 pp_size=4 kv_cache_size_num_tokens='(1024 * 1024)' max_seqs_per_forward=1024 max_tokens_per_forward=2048`:
- In total, your server will have a KV cache size of 2 million tokens (1 million for each of the data parallel replicas).
- Each replica can have 1024 sequences running at once and 2048 tokens scheduled per forwards pass.
- Note that none of these numbers are multiplies by the pipeline parallel size.

### Torch Compile

Torch compiling your model can make it faster and reduce the amount of used activation memory, allowing you to increase the KV cache size further. You can turn it on with `torch_compile=T`. The reason it's off by default is because it increases server startup time (often by a minute or two, but this can be worse the first time you run the engine on a new machine with compilation enabled). As a rough rule of thumb, turn compilation off for debugging things where fast startup is handy, but keep it on for all long-running jobs.

### Hydragen

[Hydragen](https://arxiv.org/abs/2402.05099) (AKA cascade attention, bifurcated attention) is a method for more efficiently computing attention over a batch of sequences that share a common prefix. You can turn on Hydragen with `use_hydragen=T` and tokasaurus will automatically detect shared prefixes across groups of sequences actively running. You can control the thresholds where groups will be formed with `hydragen_min_group_size` and `hydragen_min_prefix_len`, which define the minimum number of sequences in a shared prefix group, and the minimum token length of a shared prefix measured in tokens, respectively. Note that turning on Hydragen can have a slight numerical impact on your generations since we combine attention results in bfloat16.

### Misc

Here are some other server flags we didn't cover above, with their corresponding defaults:

```bash
port=10210 # The port the server listens on. Note that all data parallel replicas are accessed through the same server port.
page_size=16 # The page size for the paged KV cache.
stop_string_num_token_lookback=5 # How many tokens to look back in the sequence for when checking whether a stop string has been generated. You may need to increase this if you have very long stop strings.
stats_report_seconds=5.0 # How often server stats are printed to the console.
uvicorn_log_level="info" # The logging level for the uvicorn web server handling requests. Set this value to "warning" to disable logs being printed every time a request is finished (which can sometimes be annoying/verbose).
```

## System Design

Tokasaurus has three major components:

1. A web server that interfaces between client requests and the actual engine (see `tokasaurus/server/`).
2. A manager that handles most of the CPU-side complexity (e.g. scheduling, paged kv cache management, hydragen grouping, etc.) (see `tokasaurus/manager/`).
3. A relatively barebones model worker that runs forward passes (see `tokasaurus/model/`).

The server and manager are each their own process, with the model worker corresponding to one or more processes depending on the parallelization flags. These components communicate with each other asynchronously using queues. Importantly, the manager works to ensure that there are multiple items in the model input queue, so that the model can always be running forwards passes (i.e. the GPU can always be active) and never stall waiting for the manager to send it more work.

When data parallelism is used, each replica has its own manager process and set of model worker processes. However, all data parallel replicas share the same server process which handles load balancing.

The entry point for starting up the server and kicking off all the processes is `tokasaurus/entry.py`.


## Citation
 
```
@software{juravsky2025tokasaurus,
  title  = {Tokasaurus: An LLM Inference Engine for High-Throughput Workloads},
  author = {Jordan Juravsky and Ayush Chakravarthy and Ryan Ehrlich and Sabri Eyuboglu and Bradley Brown and Joseph Shetaye and Christopher Ré and Azalia Mirhoseini},
  url    = {https://github.com/fla-org/flash-linear-attention},
  month  = jun,
  year   = {2025}
}
```
