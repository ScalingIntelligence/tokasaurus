# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tokasaurus is a high-performance LLM inference engine designed for high-throughput workloads. It implements advanced features for efficient LLM serving including data/pipeline/tensor parallelism, paged KV caching, Hydragen optimization, and end-to-end torch compilation.

## Architecture

The system uses a three-tier architecture:
- **Web Server** (`tokasaurus/server/`): FastAPI-based HTTP server with OpenAI-compatible APIs
- **Manager** (`tokasaurus/manager/`): CPU-side orchestration handling scheduling, KV cache management, and request batching
- **Model Worker** (`tokasaurus/model/`): GPU-side execution of model forward passes

Communication between components uses async queues with protobuf-like message passing.

## Development Commands

### Running the Server
```bash
# Single GPU
toka model=meta-llama/Llama-3.2-1B-Instruct

# Multi-GPU with pipeline parallelism
toka model=meta-llama/Llama-3.1-70B-Instruct pp_size=8

# Custom configuration
toka model=<model_name> tp_size=2 max_seqs_per_forward=256
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_basic.py::test_basic

# Test with specific model
MODEL=meta-llama/Llama-3.2-1B-Instruct pytest tests/test_basic.py
```

### Testing Server Connectivity
```bash
# Basic ping test
toka-ping prompt='Hello world' max_tokens=100

# Chat mode
toka-ping prompt='Hello world' max_tokens=100 chat=True
```

## Configuration System

Uses Pydra for configuration management. All configuration is in `tokasaurus/common_types.py` as the `EngineConfig` class. Pass config options as `key=value` pairs:

```bash
toka model=<model> kv_cache_num_tokens=100000 max_tokens_per_forward=8192
```

Key configuration groups:
- Model loading: `model`, `tokenizer`, `trust_remote_code`
- Parallelism: `dp_size`, `pp_size`, `tp_size`
- Memory: `kv_cache_num_tokens`, `page_size`, `max_seqs_per_forward`
- Performance: `torch_compile`, `use_cudagraphs`, `use_hydragen`

## Code Organization

- Entry points are in `entry.py` (main), `scripts/ping.py`, `scripts/download.py`
- Core types and config in `common_types.py`
- Manager logic for scheduling and allocation in `manager/`
- Model implementations in `model/` (supports Llama and Qwen)
- HTTP endpoints in `server/endpoints.py`

## Key Implementation Details

1. **KV Cache Management**: Uses paged allocation with configurable page sizes. Implementation in `manager/allocator.py` and `model/kv_cache.py`.

2. **Scheduling**: Advanced scheduler in `manager/scheduler.py` that simulates future KV cache usage to make optimal batching decisions.

3. **Hydragen**: Shared prefix optimization in `manager/hydragen.py` that groups requests with common prefixes for efficient attention computation.

4. **Model Support**: Currently supports Llama and Qwen models. New models can be added in `model/` directory following the existing pattern.

5. **Compilation**: Supports torch.compile with dynamic shapes and CUDA graphs. Warmup phase automatically triggers recompiles.

## Common Development Tasks

### Adding a New Model
1. Create new file in `model/` directory
2. Implement the model class following `llama.py` or `qwen.py` pattern
3. Register in `model/__init__.py`
4. Update `get_model_cls()` in `model/__init__.py`

### Debugging
- Set `log_level=DEBUG` for verbose logging
- Use `statsd_addr` to enable metrics collection
- Check process logs - each component (server, manager, model) logs separately

### Performance Tuning
- Adjust `max_seqs_per_forward` and `max_tokens_per_forward` for memory/throughput tradeoff
- Enable `use_cudagraphs` for lower latency (requires warmup)
- Use `use_hydragen` for workloads with shared prefixes
- Configure `torch_compile` options in `torch_compile_config`