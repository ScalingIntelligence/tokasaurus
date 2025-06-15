#!/usr/bin/env python3
"""
Script to generate fake cartridges for testing the cartridge loading system.
Creates cartridges with realistic dimensions based on the Llama model configuration.

Usage:
    python tokasaurus/scripts/cartridges/generate_fake_cartridges.py [output_dir]

If no output_dir is provided, it will use a temporary directory
"""

import json
import torch
from torch import nn
import argparse
from pathlib import Path
from typing import Dict, Any
import yaml


def get_model_config() -> Dict[str, Any]:
    """Get a standard model configuration for testing."""
    return {
        "num_key_value_heads": 8,
        "num_attention_heads": 32,
        "head_dim": 128,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "pretrained_model_name_or_path": "meta-llama/Llama-3.2-3B-Instruct"
    }


def calculate_embed_dim(config: Dict[str, Any]) -> int:
    """Calculate embedding dimension from config."""
    return config["num_key_value_heads"] * config["head_dim"]


def generate_cartridge_data(num_tokens: int, config: Dict[str, Any], dtype: torch.dtype = torch.bfloat16, num_fixed_tokens: int = 1) -> Dict[str, nn.ParameterList]:
    """
    Generate fake cartridge data with the expected structure.
    
    Returns a dictionary with ParameterLists for trainable_keys, trainable_values, 
    fixed_keys, and fixed_values.
    """
    
    num_trainable_tokens = num_tokens - num_fixed_tokens
    num_layers = config["num_hidden_layers"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    
    # Create parameter lists for each component
    trainable_keys = nn.ParameterList()
    trainable_values = nn.ParameterList()
    fixed_keys = nn.ParameterList()
    fixed_values = nn.ParameterList()
    
    for layer_idx in range(num_layers):
        # Trainable parameters: [1, num_kv_heads, num_trainable_tokens, head_dim]
        trainable_key = torch.randn(1, num_kv_heads, num_trainable_tokens, head_dim, dtype=dtype)
        trainable_value = torch.randn(1, num_kv_heads, num_trainable_tokens, head_dim, dtype=dtype)
        
        # Fixed parameters: [1, num_kv_heads, num_fixed_tokens, head_dim]
        fixed_key = torch.randn(1, num_kv_heads, num_fixed_tokens, head_dim, dtype=dtype)
        fixed_value = torch.randn(1, num_kv_heads, num_fixed_tokens, head_dim, dtype=dtype)
        
        trainable_keys.append(nn.Parameter(trainable_key))
        trainable_values.append(nn.Parameter(trainable_value))
        fixed_keys.append(nn.Parameter(fixed_key))
        fixed_values.append(nn.Parameter(fixed_value))
    
    return {
        "trainable_keys": trainable_keys,
        "trainable_values": trainable_values,
        "fixed_keys": fixed_keys,
        "fixed_values": fixed_values
    }


def create_cartridge(
    cartridge_dir: str,
    cartridge_id: str,
    num_tokens: int,
    config: Dict[str, Any],
    page_size: int = 16,
    num_fixed_tokens: int = 1,
    dtype: torch.dtype = torch.bfloat16
) -> Path:
    """
    Create a fake cartridge with the specified parameters.
    
    Args:
        cartridge_dir: Directory to create the cartridge in
        cartridge_id: Unique identifier for the cartridge
        num_tokens: Total number of tokens in the cartridge
        config: Model configuration
        page_size: Page size for memory allocation (should divide num_tokens evenly)
        num_fixed_tokens: Number of fixed (non-trainable) tokens
        dtype: Data type for the cartridge tensors
        
    Returns:
        Path to the created cartridge directory
    """
    
    # Ensure num_tokens is aligned to page_size
    if num_tokens % page_size != 0:
        aligned_tokens = ((num_tokens + page_size - 1) // page_size) * page_size
        print(f"Warning: num_tokens ({num_tokens}) not aligned to page_size ({page_size}). Adjusting to {aligned_tokens}")
        num_tokens = aligned_tokens
    
    num_trainable_tokens = num_tokens - num_fixed_tokens
    
    cartridge_path = Path(cartridge_dir) / cartridge_id
    cartridge_path.mkdir(parents=True, exist_ok=True)
    
    # Generate cartridge data
    cartridge_data = generate_cartridge_data(num_tokens, config, dtype, num_fixed_tokens)
    
    # Save cartridge data
    torch.save(cartridge_data, cartridge_path / "cartridge.pt")
    
    # Create config.yaml that mimics wandb training config structure
    cartridge_config = {
        "_config_type": {
            "_is_type": True,
            "_module": "capsules.train",
            "_qualname": "TrainConfig"
        },
        "kv_cache_initializer": {
            "_config_type": {
                "_is_type": True,
                "_module": "capsules.kv_initialization.strategies.first_n_tokens",
                "_qualname": "KVCacheInitFromFirstNTokensOfContext.Config"
            },
            "max_tokens": num_tokens,
            "num_frozen_tokens": num_fixed_tokens,
            "target": {
                "_is_type": True,
                "_module": "capsules.kv_initialization.strategies.first_n_tokens",
                "_qualname": "KVCacheInitFromFirstNTokensOfContext"
            }
        },
        "model": {
            "_config_type": {
                "_is_type": True,
                "_module": "capsules.config",
                "_qualname": "HFModelConfig"
            },
            "pretrained_model_name_or_path": config["pretrained_model_name_or_path"],
            "model_cls": {
                "_is_type": True,
                "_module": "capsules.models.llama",
                "_qualname": "LlamaForCausalLM"
            },
            "attn_implementation": "einsum",
            "tuning_method": "custom_prefix"
        },
        "name": f"test_cartridge_{cartridge_id}",
        "device": "cuda",
        "epochs": 2,
        "lr": 0.02,
        "seed": 42
    }
    
    # Save config as YAML
    with open(cartridge_path / "config.yaml", "w") as f:
        yaml.dump(cartridge_config, f, indent=2, default_flow_style=False)
    
    print(f"Created cartridge '{cartridge_id}' at {cartridge_path}")
    print(f"  - Total tokens: {num_tokens} ({num_trainable_tokens} trainable + {num_fixed_tokens} fixed)")
    print(f"  - Layers: {config['num_hidden_layers']}")
    print(f"  - KV heads: {config['num_key_value_heads']}")
    print(f"  - Head dim: {config['head_dim']}")
    print(f"  - Blocks: {num_tokens // page_size}")
    
    # Calculate total size (trainable + fixed)
    total_elements_per_layer = (num_trainable_tokens + num_fixed_tokens) * config['num_key_value_heads'] * config['head_dim'] * 2  # keys + values
    total_elements = len(cartridge_data['trainable_keys']) * total_elements_per_layer
    element_size = cartridge_data['trainable_keys'][0].element_size()
    total_bytes = total_elements * element_size
    print(f"  - Data size: {total_elements} elements ({total_bytes} bytes)")
    
    return cartridge_path


def create_test_cartridges(output_dir: str = "./cartridges", page_size: int = 16):
    """Create a set of test cartridges with different sizes."""
    
    config = get_model_config()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define test cartridges with different sizes
    test_cartridges = [
        ("small_cartridge", 16),      # 1 block
        ("medium_cartridge", 32),     # 2 blocks  
        ("large_cartridge", 64),      # 4 blocks
        ("xlarge_cartridge", 128),    # 8 blocks
        ("test_cartridge", 32),       # For compatibility with existing tests
        ("cartA", 16),                # For compatibility with existing tests
        ("cartB", 48),                # 3 blocks
        ("cartridge_A", 16),          # For compatibility with existing tests
        ("cartridge_B", 32),          # For compatibility with existing tests
        ("my_cartridge_123", 64),     # For compatibility with existing tests
    ]
    
    print(f"Creating test cartridges in {output_path}")
    print(f"Model config: {config['num_key_value_heads']} KV heads, {config['head_dim']} head dim")
    print(f"Page size: {page_size}")
    print()
    
    created_cartridges = []
    
    for cartridge_id, num_tokens in test_cartridges:
        try:
            cartridge_path = create_cartridge(
                cartridge_dir=str(output_path),
                cartridge_id=cartridge_id,
                num_tokens=num_tokens,
                config=config,
                page_size=page_size
            )
            created_cartridges.append((cartridge_id, cartridge_path))
            print()
        except Exception as e:
            print(f"Error creating cartridge '{cartridge_id}': {e}")
            print()
    
    # Create a summary file
    summary = {
        "created_cartridges": [
            {
                "cartridge_id": cartridge_id,
                "path": str(path),
                "config_file": str(path / "config.yaml"),
                "data_file": str(path / "cartridge.pt")
            }
            for cartridge_id, path in created_cartridges
        ],
        "model_config": config,
        "page_size": page_size,
        "total_cartridges": len(created_cartridges)
    }
    
    with open(output_path / "cartridges_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Created {len(created_cartridges)} cartridges")
    print(f"Summary saved to {output_path / 'cartridges_summary.json'}")
    
    return created_cartridges


def verify_cartridge(cartridge_path: Path):
    """Verify that a cartridge was created correctly."""
    
    # Load config
    with open(cartridge_path / "config.yaml", "r") as f:
        config_data = yaml.safe_load(f)
    
    # Extract relevant info from config
    kv_init = config_data.get("kv_cache_initializer", {})
    max_tokens = kv_init.get("max_tokens", 0)
    num_fixed_tokens = kv_init.get("num_frozen_tokens", 1)
    
    # Load data
    data = torch.load(cartridge_path / "cartridge.pt", map_location="cpu", weights_only=False)
    
    # Verify structure
    required_keys = ["trainable_keys", "trainable_values", "fixed_keys", "fixed_values"]
    for key in required_keys:
        assert key in data, f"Missing '{key}' in cartridge data"
    
    trainable_keys = data["trainable_keys"]
    trainable_values = data["trainable_values"]
    fixed_keys = data["fixed_keys"]
    fixed_values = data["fixed_values"]
    
    # Verify ParameterList lengths
    config = get_model_config()  # Get expected model config
    expected_layers = config["num_hidden_layers"]
    param_lists = [trainable_keys, trainable_values, fixed_keys, fixed_values]
    param_names = ["trainable_keys", "trainable_values", "fixed_keys", "fixed_values"]
    
    for param_list, param_name in zip(param_lists, param_names):
        assert len(param_list) == expected_layers, f"{param_name} length ({len(param_list)}) doesn't match expected layers ({expected_layers})"
    
    # Verify shapes of each parameter
    num_trainable_tokens = max_tokens - num_fixed_tokens
    
    expected_trainable_shape = (1, config["num_key_value_heads"], num_trainable_tokens, config["head_dim"])
    expected_fixed_shape = (1, config["num_key_value_heads"], num_fixed_tokens, config["head_dim"])
    
    for i in range(len(trainable_keys)):
        # Check trainable parameters
        trainable_key_param = trainable_keys[i]
        trainable_value_param = trainable_values[i]
        assert trainable_key_param.shape == expected_trainable_shape, f"Layer {i} trainable key shape mismatch: {trainable_key_param.shape} vs expected {expected_trainable_shape}"
        assert trainable_value_param.shape == expected_trainable_shape, f"Layer {i} trainable value shape mismatch: {trainable_value_param.shape} vs expected {expected_trainable_shape}"
        
        # Check fixed parameters
        fixed_key_param = fixed_keys[i]
        fixed_value_param = fixed_values[i]
        assert fixed_key_param.shape == expected_fixed_shape, f"Layer {i} fixed key shape mismatch: {fixed_key_param.shape} vs expected {expected_fixed_shape}"
        assert fixed_value_param.shape == expected_fixed_shape, f"Layer {i} fixed value shape mismatch: {fixed_value_param.shape} vs expected {expected_fixed_shape}"
    
    # Verify token breakdown
    expected_total = num_trainable_tokens + num_fixed_tokens
    assert max_tokens == expected_total, f"Token mismatch: config total ({max_tokens}) != trainable + fixed ({expected_total})"
    
    print(f"✓ Cartridge with {max_tokens} tokens verified successfully")
    print(f"  Layers: {len(trainable_keys)}")
    print(f"  Trainable shape per layer: {trainable_keys[0].shape}, Dtype: {trainable_keys[0].dtype}")
    print(f"  Fixed shape per layer: {fixed_keys[0].shape}, Dtype: {fixed_keys[0].dtype}")
    print(f"  Total tokens: {max_tokens} ({num_trainable_tokens} trainable + {num_fixed_tokens} fixed)")


def main():
    parser = argparse.ArgumentParser(description="Generate fake cartridges for testing")
    parser.add_argument("--output-dir", "-o", default="./cartridges", 
                       help="Output directory for cartridges (default: ./cartridges)")
    parser.add_argument("--page-size", "-p", type=int, default=16,
                       help="Page size for cartridges (default: 16)")
    parser.add_argument("--verify", "-v", action="store_true",
                       help="Verify created cartridges after generation")
    parser.add_argument("--cartridge-id", "-c", type=str,
                       help="Create a single cartridge with this ID")
    parser.add_argument("--num-tokens", "-n", type=int, default=32,
                       help="Number of tokens for single cartridge (default: 32)")
    
    args = parser.parse_args()
    
    if args.cartridge_id:
        # Create single cartridge
        config = get_model_config()
        cartridge_path = create_cartridge(
            cartridge_dir=args.output_dir,
            cartridge_id=args.cartridge_id,
            num_tokens=args.num_tokens,
            config=config,
            page_size=args.page_size
        )
        
        if args.verify:
            verify_cartridge(cartridge_path)
    else:
        # Create test suite
        created_cartridges = create_test_cartridges(
            output_dir=args.output_dir,
            page_size=args.page_size
        )
        
        if args.verify:
            print("\nVerifying cartridges...")
            for cartridge_id, cartridge_path in created_cartridges:
                verify_cartridge(cartridge_path)


if __name__ == "__main__":
    main() 