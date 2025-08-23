#!/usr/bin/env python3

import json
import os
import requests
import tempfile
from openai import OpenAI

# Simple test script for streaming functionality
# Usage: python test_streaming.py

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")

def test_streaming():
    """Test the streaming chat completions endpoint"""
    
    # Use localhost server - adjust port as needed
    PORT = 8000  # Default port, adjust if needed
    base_url = f"http://localhost:{PORT}/v1"
    
    client = OpenAI(
        api_key="test-key", 
        base_url=base_url
    )
    
    try:
        # Test streaming request
        print("Testing streaming chat completion...")
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Say hello world"}
            ],
            max_tokens=20,
            temperature=0.0,
            stream=True
        )
        
        print("Streaming response:")
        collected_content = ""
        chunk_count = 0
        
        for chunk in response:
            chunk_count += 1
            print(f"Chunk {chunk_count}: {chunk}")
            
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content') and choice.delta.content:
                    collected_content += choice.delta.content
        
        print(f"\nCollected content: '{collected_content}'")
        print(f"Total chunks received: {chunk_count}")
        
        # Test non-streaming for comparison
        print("\nTesting non-streaming for comparison...")
        
        non_stream_response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Say hello world"}
            ],
            max_tokens=20,
            temperature=0.0,
            stream=False
        )
        
        print(f"Non-streaming response: {non_stream_response.choices[0].message.content}")
        
        # Basic validation
        assert collected_content == non_stream_response.choices[0].message.content, \
            f"Content mismatch: streaming='{collected_content}', non-streaming='{non_stream_response.choices[0].message.content}'"
        
        print("✅ Streaming test passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    test_streaming()