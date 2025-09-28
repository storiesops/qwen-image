#!/usr/bin/env python3
"""
Quick test to avoid timeouts - small image, few steps
"""
import requests
import json
import time
import base64

API_URL = "https://v9edjr0o921q6q-8000.proxy.runpod.net"

def quick_test():
    """Test with minimal parameters to avoid timeout"""
    print("=== QUICK DFloat11 TEST ===")
    
    payload = {
        "prompt": "a cute dog",
        "negative_prompt": "blurry, low quality, distorted",  # Required for CFG!
        "width": 512,  # Smaller image
        "height": 512,
        "num_inference_steps": 10,  # Much fewer steps
        "true_cfg_scale": 4.0,
        "seed": 42
    }
    
    print("Testing with minimal parameters to avoid timeout...")
    print(f"Parameters: {json.dumps(payload, indent=2)}")
    
    start_time = time.time()
    response = requests.post(f"{API_URL}/generate", json=payload, timeout=120)
    end_time = time.time()
    
    print(f"Status: {response.status_code}")
    print(f"Time: {end_time - start_time:.1f}s")
    
    if response.status_code == 200:
        result = response.json()
        print("[SUCCESS] Generation completed!")
        print(f"Image size: {len(result['image'])} chars")
        
        # Save the image
        with open("quick_test_output.png", "wb") as f:
            f.write(base64.b64decode(result['image']))
        print("[SAVED] quick_test_output.png")
        
        return True
    else:
        print(f"[FAIL] {response.status_code}")
        print(response.text)
        return False

if __name__ == "__main__":
    quick_test()
