#!/usr/bin/env python3
"""
Test script for downloading cartridges using the cartridge_downloader module.
Tests wandb, huggingface, and local source functionality.

Usage:
    python scripts/cartridges/download_cartridge.py [cartridge_id] [test_dir]

If no cartridge_id is provided, it will use the default "wauoq23f"
If no test_dir is provided, it will use a temporary directory
"""

import sys
import tempfile
import time
from pathlib import Path

from loguru import logger
from tokasaurus.manager.cartridge_downloader import download_cartridge
from tokasaurus.utils import sanitize_cartridge_id


def test_local_cartridge(cartridge_id: str = "small_cartridge", cartridge_dir: Path = Path("./cartridges")):
    """
    Test local cartridge validation.
    
    Args:
        cartridge_id: The cartridge ID to test (default: "small_cartridge")
        cartridge_dir: Directory where cartridges are stored
    """
    logger.info(f"Testing local cartridge validation: {cartridge_id}")
    
    try:
        # Test local source validation
        download_cartridge(
            cartridge_id=cartridge_id,
            source="local",
            cartridges_path=cartridge_dir,
            force_redownload=False  # This should be ignored for local sources
        )
        
        logger.info("✅ Local cartridge validation successful!")
        return True
        
    except FileNotFoundError as e:
        logger.error(f"❌ Local cartridge not found: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Local cartridge test failed: {e}")
        return False


def test_download_cartridge(cartridge_id: str = "wauoq23f", test_dir: Path | None = None):
    """
    Test downloading a cartridge from wandb and verify its existence.
    
    Args:
        cartridge_id: The cartridge ID to download (default: "wauoq23f")
        test_dir: Directory to download to (if None, uses temp directory)
    """
    if test_dir is None:
        test_dir = Path(tempfile.mkdtemp())
        logger.info(f"Using temporary directory: {test_dir}")
    else:
        test_dir = Path(test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Testing download of cartridge: {cartridge_id}")
    
    try:
        # Test downloading from wandb
        download_cartridge(
            cartridge_id=cartridge_id,
            source="wandb",
            cartridges_path=test_dir,
            force_redownload=False
        )
        
        # Verify the cartridge was downloaded
        cartridge_dir = test_dir / cartridge_id
        cartridge_file = cartridge_dir / "cartridge.pt"
        config_file = cartridge_dir / "config.yaml"
        
        if not cartridge_file.exists():
            raise FileNotFoundError(f"Cartridge file not found: {cartridge_file}")
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        # Check file sizes
        cartridge_size = cartridge_file.stat().st_size
        config_size = config_file.stat().st_size
        
        logger.info(f"✅ Download successful!")
        logger.info(f"   Cartridge file: {cartridge_file} ({cartridge_size / (1024*1024):.2f} MB)")
        logger.info(f"   Config file: {config_file} ({config_size} bytes)")
        
        # Test force redownload - capture original timestamps first
        logger.info("Testing force redownload...")
        original_cartridge_mtime = cartridge_file.stat().st_mtime
        original_config_mtime = config_file.stat().st_mtime
        
        # Sleep briefly to ensure timestamps would be different if files are re-downloaded
        time.sleep(0.1)
        
        download_cartridge(
            cartridge_id=cartridge_id,
            source="wandb",
            cartridges_path=test_dir,
            force_redownload=True
        )
        
        # Verify files were actually re-downloaded by checking timestamps
        new_cartridge_mtime = cartridge_file.stat().st_mtime
        new_config_mtime = config_file.stat().st_mtime
        
        if new_cartridge_mtime <= original_cartridge_mtime:
            raise AssertionError("Cartridge file was not re-downloaded during force redownload!")
        
        if new_config_mtime <= original_config_mtime:
            raise AssertionError("Config file was not re-downloaded during force redownload!")
        
        logger.info("✅ Force redownload successful - files were actually re-downloaded!")
        logger.info(f"   Cartridge file timestamp changed: {original_cartridge_mtime:.3f} -> {new_cartridge_mtime:.3f}")
        logger.info(f"   Config file timestamp changed: {original_config_mtime:.3f} -> {new_config_mtime:.3f}")
        
        # Test skipping existing download - capture timestamps again
        logger.info("Testing skip existing download...")
        skip_test_cartridge_mtime = cartridge_file.stat().st_mtime
        skip_test_config_mtime = config_file.stat().st_mtime
        
        # Sleep briefly to ensure timestamps would be different if files are re-downloaded
        time.sleep(0.1)
        
        download_cartridge(
            cartridge_id=cartridge_id,
            source="wandb",
            cartridges_path=test_dir,
            force_redownload=False
        )
        
        # Verify files were NOT re-downloaded when force_redownload=False
        final_cartridge_mtime = cartridge_file.stat().st_mtime
        final_config_mtime = config_file.stat().st_mtime
        
        if final_cartridge_mtime != skip_test_cartridge_mtime:
            raise AssertionError("Cartridge file was unexpectedly re-downloaded when force_redownload=False!")
        
        if final_config_mtime != skip_test_config_mtime:
            raise AssertionError("Config file was unexpectedly re-downloaded when force_redownload=False!")
        
        logger.info("✅ Skip existing download successful - files were correctly NOT re-downloaded!")
        logger.info(f"   Cartridge file timestamp unchanged: {skip_test_cartridge_mtime:.3f}")
        logger.info(f"   Config file timestamp unchanged: {skip_test_config_mtime:.3f}")
        
        # Now test local validation on the downloaded cartridge
        logger.info("Testing local validation on downloaded cartridge...")
        download_cartridge(
            cartridge_id=cartridge_id,
            source="local",
            cartridges_path=test_dir,
            force_redownload=False  # Should be ignored for local
        )
        logger.info("✅ Local validation successful!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


def test_download_cartridge_huggingface(cartridge_id: str = "hazyresearch/cartridge-wauoq23f", test_dir: Path | None = None):
    """
    Test downloading a cartridge from HuggingFace and verify its existence.
    
    Args:
        cartridge_id: The HuggingFace cartridge ID to download (default: "hazyresearch/cartridge-wauoq23f")
        test_dir: Directory to download to (if None, uses temp directory)
    """
    if test_dir is None:
        test_dir = Path(tempfile.mkdtemp())
        logger.info(f"Using temporary directory: {test_dir}")
    else:
        test_dir = Path(test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Testing download of HuggingFace cartridge: {cartridge_id}")
    
    try:
        # Test downloading from HuggingFace
        download_cartridge(
            cartridge_id=cartridge_id,
            source="huggingface",
            cartridges_path=test_dir,
            force_redownload=False
        )
        
        # Verify the cartridge was downloaded
        # Use the sanitized cartridge ID to get the correct directory name
        sanitized_cartridge_id = sanitize_cartridge_id(cartridge_id)
        cartridge_dir = test_dir / sanitized_cartridge_id
        cartridge_file = cartridge_dir / "cartridge.pt"
        config_file = cartridge_dir / "config.yaml"
        
        if not cartridge_file.exists():
            raise FileNotFoundError(f"Cartridge file not found: {cartridge_file}")
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        # Check file sizes
        cartridge_size = cartridge_file.stat().st_size
        config_size = config_file.stat().st_size
        
        logger.info(f"✅ HuggingFace download successful!")
        logger.info(f"   Cartridge file: {cartridge_file} ({cartridge_size / (1024*1024):.2f} MB)")
        logger.info(f"   Config file: {config_file} ({config_size} bytes)")
        
        # Test force redownload - capture original timestamps first
        logger.info("Testing force redownload from HuggingFace...")
        original_cartridge_mtime = cartridge_file.stat().st_mtime
        original_config_mtime = config_file.stat().st_mtime
        
        # Sleep briefly to ensure timestamps would be different if files are re-downloaded
        time.sleep(0.1)
        
        download_cartridge(
            cartridge_id=cartridge_id,
            source="huggingface",
            cartridges_path=test_dir,
            force_redownload=True
        )

        return True
        
    except Exception as e:
        logger.error(f"❌ HuggingFace test failed: {e}")
        return False

def test_download_cartridge_s3(
    cartridge_id: str = "s3://engram-cartridges/weights/cartridge-llama-40455a296d32046e/cache-step27.pt",
    test_dir: Path | None = None,
):
    """
    Test downloading a cartridge from S3 and verify its existence.
    
    Args:
        cartridge_id: The S3 cartridge ID to download (default: "s3://engram-cartridges/weights/cartridge-llama-40455a296d32046e/cache-step27.pt")
        test_dir: Directory to download to (if None, uses temp directory)
    """
    if test_dir is None:
        test_dir = Path(tempfile.mkdtemp())
        logger.info(f"Using temporary directory: {test_dir}")

    from tokasaurus.manager.cartridge_downloader import validate_cartridge_exists
    validate_cartridge_exists(cartridge_id, source="s3")
    
    logger.info(f"Testing S3 cartridge download: {cartridge_id}")
    
    try:
        # Test downloading from S3
        download_cartridge(
            cartridge_id=cartridge_id,
            source="s3",
            cartridges_path=test_dir,
            force_redownload=True
        )
        
        # Verify the cartridge was downloaded
        # Use the sanitized cartridge ID to get the correct directory name
        sanitized_cartridge_id = sanitize_cartridge_id(cartridge_id)
        cartridge_dir = test_dir / sanitized_cartridge_id
        
        # For S3 downloads, the file might be downloaded directly as a .pt file
        # Check for both cartridge.pt and the original filename
        cartridge_file = cartridge_dir / "cartridge.pt"
        if not cartridge_file.exists():
            # Try the original filename
            original_filename = Path(cartridge_id).name
            cartridge_file = cartridge_dir / original_filename
        
        if not cartridge_file.exists():
            raise FileNotFoundError(f"Cartridge file not found in: {cartridge_dir}")
        
        # Check file size
        cartridge_size = cartridge_file.stat().st_size
        
        logger.info(f"✅ S3 download successful!")
        logger.info(f"   Cartridge file: {cartridge_file} ({cartridge_size / (1024*1024):.2f} MB)")
        
        # Test force redownload - capture original timestamp first
        logger.info("Testing force redownload from S3...")
        original_cartridge_mtime = cartridge_file.stat().st_mtime
        
        # Sleep briefly to ensure timestamps would be different if files are re-downloaded
        time.sleep(0.1)
        
        download_cartridge(
            cartridge_id=cartridge_id,
            source="s3",
            cartridges_path=test_dir,
            force_redownload=True
        )
        
        # Verify the file was re-downloaded (timestamp should be different)
        new_cartridge_mtime = cartridge_file.stat().st_mtime
        if new_cartridge_mtime > original_cartridge_mtime:
            logger.info("✅ Force redownload from S3 successful!")
        else:
            logger.warning("⚠️ Force redownload may not have updated the file timestamp")

        from tokasaurus.manager.types import CartridgeConfig
        config_file = cartridge_dir / "config.yaml"
        out = CartridgeConfig.from_config_yaml(cartridge_id, config_file)

        import torch
        state_dict = torch.load(cartridge_file, map_location="cpu", weights_only=False)
        from tokasaurus.model.types import _convert_from_torchtitan
        state_dict = _convert_from_torchtitan(state_dict)
        breakpoint()
        logger.info(f"Cartridge config: {out}")
        return True
        
    except Exception as e:
        logger.error(f"❌ S3 test failed: {e}")
        return False

def test_invalid_source():
    """Test invalid source handling."""
    logger.info("Testing invalid source handling...")
    
    try:
        download_cartridge(
            cartridge_id="test_cartridge",
            source="invalid_source",
            cartridges_path=Path("./cartridges"),
            force_redownload=False
        )
        logger.error("❌ Invalid source should have been rejected!")
        return False
    except ValueError as e:
        if "Unsupported cartridge source" in str(e):
            logger.info("✅ Invalid source properly rejected!")
            return True
        else:
            logger.error(f"❌ Unexpected error message: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ Unexpected exception: {e}")
        return False

if __name__ == "__main__":
    # Get cartridge ID from command line or use default
    cartridge_id = sys.argv[1] if len(sys.argv) > 1 else "wauoq23f"
    
    # Get test directory from command line if provided
    test_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    logger.info("Starting cartridge downloader tests...")
    
    all_tests_passed = True
    
    # # Test 1: Local cartridge validation
    # logger.info("\n" + "="*50)
    # logger.info("TEST 1: Local Cartridge Validation")
    # logger.info("="*50)
    # if not test_local_cartridge():
    #     all_tests_passed = False
    
    # # Test 2: Wandb download and validation
    # logger.info("\n" + "="*50)
    # logger.info("TEST 2: Wandb Download and Validation")
    # logger.info("="*50)
    # if not test_download_cartridge(cartridge_id, test_dir):
    #     all_tests_passed = False
    
    # # Test 3: HuggingFace download and validation
    # logger.info("\n" + "="*50)
    # logger.info("TEST 3: HuggingFace Download and Validation")
    # logger.info("="*50)
    # if not test_download_cartridge_huggingface("hazyresearch/cartridge-" + cartridge_id, test_dir):
    #     all_tests_passed = False
    
    # Test 4: S3 download and validation
    logger.info("\n" + "="*50)
    logger.info("TEST 4: S3 Download and Validation")
    logger.info("="*50)
    if not test_download_cartridge_s3(
        "s3://engram-cartridges/weights/torchtitan/run-2025-11-23-a5193062/step-6/model.pt",
        test_dir
    ):
        all_tests_passed = False
    
    # Test 5: Invalid source handling
    logger.info("\n" + "="*50)
    logger.info("TEST 5: Invalid Source Handling")
    logger.info("="*50)
    if not test_invalid_source():
        all_tests_passed = False
    
    logger.info("\n" + "="*50)
    if all_tests_passed:
        logger.info("🎉 All tests passed!")
    else:
        logger.error("💥 Some tests failed!")
        sys.exit(1) 