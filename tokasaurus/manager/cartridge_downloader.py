#!/usr/bin/env python3
import json
import os
from pathlib import Path

import torch
import wandb
import yaml
from loguru import logger
from transformers import AutoConfig

from tokasaurus.utils import sanitize_cartridge_id

WANDB_PROJECT_ID = "cartridges"

def _verify_cartridge_data(cartridge_file: Path, logger):
    """
    Common function to verify cartridge data structure and log dimensions.
    
    Args:
        cartridge_file: Path to the cartridge.pt file
        logger: Logger instance to use
    """
    logger.info("Loading cartridge to verify dimensions...")
    cartridge_data = torch.load(cartridge_file, map_location="cpu", weights_only=False)

    # Verify cartridge structure but don't need to create additional metadata
    # since we now use the original config.yaml
    if "trainable_keys" in cartridge_data and "fixed_keys" in cartridge_data:
        trainable_keys = cartridge_data["trainable_keys"]
        fixed_keys = cartridge_data["fixed_keys"]

        if len(trainable_keys) > 0 and len(fixed_keys) > 0:
            trainable_shape = trainable_keys[0].shape
            fixed_shape = fixed_keys[0].shape
            logger.info(f"Actual trainable shape: {trainable_shape}")
            logger.info(f"Actual fixed shape: {fixed_shape}")

            if len(trainable_shape) == 4 and len(fixed_shape) == 4:
                _, actual_num_kv_heads, num_trainable_tokens, actual_head_dim = trainable_shape
                _, _, num_fixed_tokens, _ = fixed_shape
                actual_num_layers = len(trainable_keys)

                total_tokens = num_trainable_tokens + num_fixed_tokens
                logger.info(f"Verified cartridge: {total_tokens} total tokens ({num_trainable_tokens} trainable + {num_fixed_tokens} fixed)")


def _clean_yaml_config(config_file: Path, logger):
    """
    Clean a YAML config file by converting Python-specific YAML tags to basic YAML.
    This converts things like !!python/tuple to regular lists so they can be loaded with yaml.safe_load().
    
    Args:
        config_file: Path to the config.yaml file to clean
        logger: Logger instance to use
    """
    try:
        # First, try to load with safe_load to see if it's already clean
        with open(config_file, 'r') as f:
            yaml.safe_load(f)
        logger.info(f"Config file {config_file} is already clean YAML")
        return
    except yaml.constructor.ConstructorError:
        # File contains Python-specific tags, need to clean it
        logger.info(f"Cleaning Python-specific YAML tags in {config_file}")
        
        # Load with full loader that can handle Python constructs
        with open(config_file, 'r') as f:
            try:
                config_data = yaml.load(f, Loader=yaml.FullLoader)
            except Exception as e:
                logger.warning(f"Could not clean YAML file {config_file} with FullLoader, trying UnsafeLoader: {e}")
                f.seek(0)
                config_data = yaml.load(f, Loader=yaml.UnsafeLoader)
        
        # Save back as clean YAML
        with open(config_file, 'w') as f:
            yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Successfully cleaned config file {config_file}")
    except Exception as e:
        logger.warning(f"Unexpected error while checking/cleaning YAML file {config_file}: {e}")


def download_cartridge_from_wandb(cartridge_id: str, cartridges_path: Path, force_redownload: bool = False, logger=None):
    """
    Downloads a cartridge from a wandb run.
    It fetches the run, finds the .pt file, downloads it,
    and downloads the config.yaml file from the wandb run.
    
    Args:
        cartridge_id: The wandb run ID to download
        cartridges_path: The base path where cartridges are stored
        force_redownload: If True, redownload even if cartridge already exists locally
        logger: Logger instance to use (if None, uses global logger)
    """
    if logger is None:
        from loguru import logger as global_logger
        logger = global_logger
        
    # Sanitize cartridge_id for safe directory creation
    sanitized_id = sanitize_cartridge_id(cartridge_id)
    cartridge_dir = cartridges_path / sanitized_id
    cartridge_file = cartridge_dir / "cartridge.pt"
    config_file = cartridge_dir / "config.yaml"
    
    # Check if cartridge already exists and skip if not force redownload
    if not force_redownload and cartridge_file.exists() and config_file.exists():
        logger.info(f"Cartridge {cartridge_id} already exists locally, skipping download")
        return
    
    api = wandb.Api()

    try:
        # NOTE: hardcoded project. This might need to be configurable in the future.
        # Use original cartridge_id for API call, not sanitized version
        run = api.run(f"hazy-research/{WANDB_PROJECT_ID}/{cartridge_id}")
    except wandb.errors.CommError as e:
        logger.error(f"Could not find wandb run for cartridge_id: hazy-research/{WANDB_PROJECT_ID}/{cartridge_id}. Error: {e}")
        raise FileNotFoundError(f"Could not find wandb run for cartridge_id: hazy-research/{WANDB_PROJECT_ID}/{cartridge_id}") from e

    # Find the cartridge file
    pt_files = [f for f in run.files() if f.name.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError(f"No .pt file found in wandb run {cartridge_id}")
    if len(pt_files) > 1:
        # Use most recently modified file
        pt_files.sort(key=lambda f: f.updated_at, reverse=True)
        logger.warning(f"Multiple .pt files found in wandb run {cartridge_id}, using most recent file: {pt_files[0].name}")

    wandb_file_name = pt_files[0].name
    if cartridge_id == "2o672jhw":
        wandb_file_name = "cache-step512.pt"

    # Find the config.yaml file
    config_files = [f for f in run.files() if f.name == "config.yaml"]
    if not config_files:
        raise FileNotFoundError(f"No config.yaml file found in wandb run {cartridge_id}")

    cartridge_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {wandb_file_name} from run {cartridge_id}...")
    logger.info(f"Saving to {cartridge_file}")

    # Download the cartridge file
    temp_download_path = cartridge_dir / wandb_file_name
    run.file(wandb_file_name).download(root=cartridge_dir, replace=True)
    os.rename(temp_download_path, cartridge_file)

    logger.info(f"Successfully downloaded and moved to {cartridge_file}")

    # Download the config.yaml file
    logger.info(f"Downloading config.yaml from run {cartridge_id}...")
    run.file("config.yaml").download(root=cartridge_dir, replace=True)
    logger.info(f"Successfully downloaded config.yaml to {config_file}")

    # Clean the config.yaml file to remove Python-specific YAML tags
    _clean_yaml_config(config_file, logger)

    size = os.path.getsize(cartridge_file)
    logger.info(f"File size: {size / (1024*1024):.2f} MB")

    _verify_cartridge_data(cartridge_file, logger)

    logger.info(f"Cartridge {cartridge_id} download completed successfully")


def download_cartridge_from_huggingface(cartridge_id: str, cartridges_path: Path, force_redownload: bool = False, logger=None):
    """
    Downloads a cartridge from HuggingFace Hub.
    
    Args:
        cartridge_id: The HuggingFace repository ID (e.g., "hazyresearch/cartridge-wauoq23f")
        cartridges_path: The base path where cartridges are stored
        force_redownload: If True, redownload even if cartridge already exists locally
        logger: Logger instance to use (if None, uses global logger)
    """
    if logger is None:
        from loguru import logger as global_logger
        logger = global_logger
    
    # Sanitize cartridge_id for safe directory creation
    sanitized_id = sanitize_cartridge_id(cartridge_id)
    cartridge_dir = cartridges_path / sanitized_id
    cartridge_file = cartridge_dir / "cartridge.pt"
    config_file = cartridge_dir / "config.yaml"
    
    # Check if cartridge already exists and skip if not force redownload
    if not force_redownload and cartridge_file.exists() and config_file.exists():
        logger.info(f"Cartridge {cartridge_id} already exists locally, skipping download")
        return
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        error_msg = "huggingface_hub is not installed. Please install it with: pip install huggingface_hub"
        logger.error(error_msg)
        raise ImportError(error_msg)
    
    try:
        # First, do a dry run to check what files are available without downloading
        from huggingface_hub import list_repo_files
        # Use original cartridge_id for API call, not sanitized version
        repo_files = list_repo_files(cartridge_id)
        
        # Find the cartridge file
        pt_files = [f for f in repo_files if f.endswith(".pt")]
        if not pt_files:
            raise FileNotFoundError(f"No .pt file found in HuggingFace repository {cartridge_id}")
        if len(pt_files) > 1:
            logger.warning(f"Multiple .pt files found in HuggingFace repository {cartridge_id}, will use first found: {pt_files}")
        
        # Find the config.yaml file
        if "config.yaml" not in repo_files:
            raise FileNotFoundError(f"No config.yaml file found in HuggingFace repository {cartridge_id}")
        
    except Exception as e:
        logger.error(f"Could not access HuggingFace repository {cartridge_id}. Error: {e}")
        raise FileNotFoundError(f"Could not access HuggingFace repository {cartridge_id}") from e
    
    cartridge_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {cartridge_id} from HuggingFace Hub...")
    logger.info(f"Saving to {cartridge_dir}")
    
    # Download the repository
    # Use original cartridge_id for API call, not sanitized version
    downloaded_path = snapshot_download(
        repo_id=cartridge_id,
        local_dir=cartridge_dir,
        local_dir_use_symlinks=False,  # Download actual files, not symlinks
    )
    
    # Find and standardize the cartridge file name
    downloaded_pt_files = list(cartridge_dir.glob("*.pt"))
    if len(downloaded_pt_files) == 0:
        raise FileNotFoundError(f"No .pt file found in HuggingFace repository {cartridge_id}")

    original_pt_file = downloaded_pt_files[0]
    if len(downloaded_pt_files) > 1:
        logger.warning(f"Multiple .pt files found in HuggingFace repository {cartridge_id}: {downloaded_pt_files}\nUsing the first one: {original_pt_file}")
    
    # Rename to standard name if not already named correctly
    if original_pt_file.name != "cartridge.pt":
        logger.info(f"Renaming {original_pt_file.name} to cartridge.pt for consistency")
        original_pt_file.rename(cartridge_file)
    
    logger.info(f"Successfully downloaded and moved to {cartridge_file}")
    
    # Verify config.yaml was downloaded
    if not config_file.exists():
        raise FileNotFoundError(f"config.yaml not found in downloaded cartridge {cartridge_id}")
    
    logger.info(f"Successfully downloaded config.yaml to {config_file}")
    
    # Clean the config.yaml file to remove Python-specific YAML tags
    _clean_yaml_config(config_file, logger)

    size = os.path.getsize(cartridge_file)
    logger.info(f"File size: {size / (1024*1024):.2f} MB")
    
    _verify_cartridge_data(cartridge_file, logger)
    
    logger.info(f"Cartridge {cartridge_id} download completed successfully")


def download_cartridge(cartridge_id: str, source: str, cartridges_path: Path, force_redownload: bool = False, logger=None):
    """
    Downloads a cartridge from the specified source.
    
    Args:
        cartridge_id: The cartridge ID to download
        source: The source to download from ('wandb', 'local', 'huggingface')
        cartridges_path: The base path where cartridges are stored
        force_redownload: If True, redownload even if cartridge already exists locally
        logger: Logger instance to use (if None, uses global logger)
    """
    if logger is None:
        from loguru import logger as global_logger
        logger = global_logger
        
    match source:
        case "wandb":
            download_cartridge_from_wandb(cartridge_id, cartridges_path, force_redownload, logger)
        case "local":
            # For local source, just verify the cartridge exists
            # Sanitize cartridge_id for safe directory access
            sanitized_id = sanitize_cartridge_id(cartridge_id)
            cartridge_dir = cartridges_path / sanitized_id
            cartridge_file = cartridge_dir / "cartridge.pt"
            config_file = cartridge_dir / "config.yaml"
            
            if not cartridge_file.exists() or not config_file.exists():
                raise FileNotFoundError(f"Local cartridge '{cartridge_id}' not found at {cartridge_dir}. Expected files: cartridge.pt, config.yaml")
            
            logger.info(f"Local cartridge '{cartridge_id}' found at {cartridge_dir}")
        case "huggingface":
            download_cartridge_from_huggingface(cartridge_id, cartridges_path, force_redownload, logger)
        case _:
            raise ValueError(f"Unsupported cartridge source: {source}") 


def validate_cartridge_exists(cartridge_id: str, source: str, logger=None):
    """
    Quick validation to check if a cartridge exists without downloading it.
    Raises appropriate exceptions if the cartridge cannot be found.
    
    Args:
        cartridge_id: The cartridge ID to validate
        source: The source to validate against ('wandb', 'local', 'huggingface')
        logger: Logger instance to use (if None, uses global logger)
    """
    if logger is None:
        from loguru import logger as global_logger
        logger = global_logger
        
    match source:
        case "wandb":
            # Quick check if wandb run exists
            api = wandb.Api()
            try:
                # Use original cartridge_id for API call
                print(f"Checking if cartridge {cartridge_id} exists in wandb")
                run = api.run(f"hazy-research/{WANDB_PROJECT_ID}/{cartridge_id}")
                # Just check if we can access the run - don't download anything
                logger.debug(f"Cartridge {cartridge_id} exists in wandb")
            except wandb.errors.CommError as e:
                logger.error(f"Could not find wandb run for cartridge_id: hazy-research/{WANDB_PROJECT_ID}/{cartridge_id}. Error: {e}")
                raise FileNotFoundError(f"Could not find wandb run for cartridge_id: hazy-research/{WANDB_PROJECT_ID}/{cartridge_id}") from e
                
        case "huggingface":
            # Quick check if HuggingFace repository exists
            try:
                from huggingface_hub import list_repo_files
                # Use original cartridge_id for API call
                repo_files = list_repo_files(cartridge_id)
                logger.debug(f"Cartridge {cartridge_id} exists in HuggingFace")
            except ImportError:
                error_msg = "huggingface_hub is not installed. Please install it with: pip install huggingface_hub"
                logger.error(error_msg)
                raise ImportError(error_msg)
            except Exception as e:
                logger.error(f"Could not access HuggingFace repository {cartridge_id}. Error: {e}")
                raise FileNotFoundError(f"Could not access HuggingFace repository {cartridge_id}") from e
                
        case "local":
            # For local source, just verify the cartridge exists
            sanitized_id = sanitize_cartridge_id(cartridge_id)
            # This will raise appropriate errors if paths don't exist
            logger.debug(f"Local cartridge validation for {cartridge_id} not implemented in validate_cartridge_exists")
            
        case _:
            raise ValueError(f"Unsupported cartridge source: {source}") 