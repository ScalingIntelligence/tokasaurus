# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install from PyPI
pip install tokasaurus

# Install from source (development)
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Running the Engine
```bash
# Basic engine launch (default port 10210)
toka model=meta-llama/Llama-3.2-1B-Instruct

# Test the running engine
toka-ping prompt='tell me a joke' max_tokens=256 chat=True

# Launch with multiple GPUs (pipeline parallelism)
toka model=meta-llama/Llama-3.1-70B-Instruct kv_cache_num_tokens='(512 * 1024)' pp_size=8
```

### Testing
```bash
# Run all tests
pytest

# Run specific test (e.g., for new model integration)
pytest tests/test_logprobs.py -k test_logprobs -s

# Test with specific model
MODEL=Qwen/Qwen3-0.6B pytest tests/test_logprobs.py -k test_logprobs -s

# Verbose test output for debugging
pytest --capture=no -s
```

### Type Checking
```bash
# Run type checking with pyright
pyright
```

### Benchmarking
```bash
# Run benchmarks (located in tokasaurus/benchmarks/)
python tokasaurus/benchmarks/bench_model.py
python tokasaurus/benchmarks/monkeys_gsm8k.py
```

## Architecture Overview

Tokasaurus is a high-throughput LLM inference engine with a multi-process architecture:

### Core Components
1. **Web Server** (`tokasaurus/server/`): FastAPI-based server handling OpenAI-compatible API requests
2. **Manager** (`tokasaurus/manager/`): CPU-side orchestration including scheduling, KV cache management, and Hydragen grouping
3. **Model Workers** (`tokasaurus/model/`): GPU processes running forward passes

### Key Architectural Patterns
- **Multi-process design**: Server, manager, and model workers run as separate processes communicating via queues
- **Data parallelism**: Multiple replicas share the same server process with load balancing
- **Pipeline parallelism**: Model layers distributed across multiple GPUs
- **Tensor parallelism**: Including AsyncTP support
- **Paged KV caching**: With prefix caching for efficiency
- **Hydragen**: Efficient attention over shared prefixes with automatic detection

### Entry Points
- Main entry: `tokasaurus/entry.py` - starts all processes
- CLI scripts: `toka`, `toka-ping`, `toka-download` (defined in pyproject.toml)

## Model Integration

### Supported Architectures
- **Llama3**: `tokasaurus/model/llama.py`
- **Qwen2**: `tokasaurus/model/qwen.py`  
- **Qwen3**: `tokasaurus/model/qwen3.py`

### Adding New Models
1. Create model file in `tokasaurus/model/` following existing patterns
2. Must be compatible with `BatchState` interface
3. Must use `tokasaurus_attention` (FlashInfer integration)
4. Register in `tokasaurus/model/utils.py` by adding to `model_type` union and `MODEL_MAPPING`
5. Test with: `MODEL=your-model pytest tests/test_logprobs.py -k test_logprobs -s`

## Configuration System

Uses [Pydra](https://github.com/jordan-benjamin/pydra) with `key=value` format:
- Boolean shortcuts: `key=T` (True), `key=F` (False)
- Python expressions: `key='(2 * 1024)'`
- Required: `model` (HuggingFace repo or local path)
- Optional: `tokenizer` (defaults to model path)

### Key Configuration Flags
- **Parallelism**: `dp_size`, `pp_size`, `tp_size`
- **Memory**: `kv_cache_size_num_tokens`, `max_tokens_per_forward`, `max_seqs_per_forward`
- **Performance**: `torch_compile=T`, `use_hydragen=T`
- **Server**: `port=10210`, `uvicorn_log_level="info"`

## Development Guidelines

### Code Structure
- Model implementations follow HuggingFace transformers patterns
- Use existing utilities in `tokasaurus/model/attention_utils.py`
- Follow type hints defined in `tokasaurus/common_types.py`
- Manager types in `tokasaurus/manager/types.py`

### Testing Requirements
- All model integrations must pass `test_logprobs.py`
- Requires `transformers >= 4.51.0` for Qwen3 support
- Use verbose flags for debugging: `--capture=no -s`

### PyRight Configuration
- `reportOptionalMemberAccess`: disabled
- `reportPossiblyUnboundVariable`: disabled