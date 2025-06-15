#!/usr/bin/env python3
"""
Minimal example demonstrating cartridge usage with the Tokasaurus API.

This example shows how to:
1. Make requests without cartridges (baseline)
2. Load cartridges from wandb
3. Load cartridges from HuggingFace
4. Handle basic errors

Usage:
    python tokasaurus/scripts/cartridges/example.py
"""

import requests
import json
from typing import List, Dict, Any, Optional

# API endpoint
BASE_URL = "http://localhost:10210/v1/cartridge/chat/completions"

def send_request(messages: List[Dict[str, str]], 
                cartridges: Optional[List[Dict[str, Any]]] = None,
                max_tokens: int = 50) -> Dict[str, Any]:
    """Send a request to the cartridge API."""
    
    payload = {
        "model": "test-model",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    # Add cartridges if provided
    if cartridges:
        payload["cartridges"] = cartridges
    
    try:
        response = requests.post(BASE_URL, json=payload, headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
            
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Could not connect to server. Make sure it's running on localhost:10210"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def print_response(result: Dict[str, Any], test_name: str):
    """Print the response in a readable format."""
    print(f"\n🧪 {test_name}")
    print("-" * 50)
    
    if result["success"]:
        data = result["data"]
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0]["message"]["content"]
            print(f"✅ Success: {content}")
        else:
            print(f"✅ Success: {data}")
    else:
        print(f"❌ Error: {result['error']}")

def main():
    """Run cartridge usage examples."""
    
    print("🚀 Cartridge Usage Examples")
    print("=" * 60)
    print("This example demonstrates basic cartridge loading patterns.")
    print(f"Server: {BASE_URL}\n")
    
    # Example 1: No cartridge (baseline)
    print("\n📋 Example 1: Request without cartridge")
    result = send_request(
        messages=[{"role": "user", "content": "Hello, how are you?"}]
    )
    print_response(result, "Baseline (no cartridge)")
    
    # Example 2: Wandb cartridge (auto-download)
    print("\n📋 Example 2: Load cartridge from wandb")
    result = send_request(
        messages=[{"role": "user", "content": "What can you tell me?"}],
        cartridges=[{
            "id": "wauoq23f",
            "source": "wandb", 
            "force_redownload": False
        }]
    )
    print_response(result, "Wandb cartridge (auto-download)")
    
    # Example 3: HuggingFace cartridge (if available)
    print("\n📋 Example 3: Load cartridge from HuggingFace")
    result = send_request(
        messages=[{"role": "user", "content": "Help me understand this."}],
        cartridges=[{
            "id": "hazyresearch/cartridge-wauoq23f",
            "source": "huggingface",
            "force_redownload": False
        }]
    )
    print_response(result, "HuggingFace cartridge")
    
    # Example 4: Force redownload
    print("\n📋 Example 4: Force redownload from wandb")
    result = send_request(
        messages=[{"role": "user", "content": "Fresh download test."}],
        cartridges=[{
            "id": "wauoq23f",
            "source": "wandb",
            "force_redownload": True  # This will redownload even if cached
        }]
    )
    print_response(result, "Force redownload from wandb")
    
    # Example 5: Error case - non-existent cartridge
    print("\n📋 Example 5: Error handling (non-existent cartridge)")
    result = send_request(
        messages=[{"role": "user", "content": "This should fail."}],
        cartridges=[{
            "id": "nonexistent_cartridge",
            "source": "wandb",
            "force_redownload": False
        }]
    )
    print_response(result, "Non-existent cartridge (error case)")

if __name__ == "__main__":
    main()
