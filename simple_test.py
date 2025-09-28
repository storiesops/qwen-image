#!/usr/bin/env python3
"""
Simple test for Qwen-Image API - avoids encoding issues
"""
import requests
import json
import time
import base64

API_URL = "https://v9edjr0o921q6q-8000.proxy.runpod.net"

def test_api():
    """Test API with simple calls"""
    print("=== QWEN-IMAGE API TEST ===")
    
    # Test 1: Root endpoint
    try:
        print("\n[1] Testing root endpoint...")
        response = requests.get(f"{API_URL}/")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("[OK] Root endpoint working")
        else:
            print("[FAIL] Root endpoint failed")
    except Exception as e:
        print(f"[FAIL] Root test failed: {e}")
        return
    
    # Test 2: Health endpoint
    try:
        print("\n[2] Testing health endpoint...")
        response = requests.get(f"{API_URL}/health")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("[OK] Health endpoint working")
            print(f"Model status: {data.get('status', 'unknown')}")
            if 'gpu_memory_used' in data:
                print(f"GPU Memory: {data['gpu_memory_used']} / {data.get('gpu_memory_total', 'unknown')}")
        else:
            print("[FAIL] Health endpoint failed")
    except Exception as e:
        print(f"[FAIL] Health test failed: {e}")
        return
    
    # Test 3: Simple generation
    try:
        print("\n[3] Testing image generation...")
        payload = {
            "prompt": "A beautiful landscape with mountains",
            "width": 512,
            "height": 512,
            "num_inference_steps": 20,
            "true_cfg_scale": 4.0,  # Official DFloat11 parameter
            "seed": 42
        }
        
        print("Sending generation request...")
        start_time = time.time()
        response = requests.post(f"{API_URL}/generate", json=payload, timeout=180)
        end_time = time.time()
        
        print(f"Status: {response.status_code}")
        print(f"Time: {end_time - start_time:.1f}s")
        
        if response.status_code == 200:
            result = response.json()
            if 'image' in result and result['image']:
                print("[SUCCESS] Generation successful!")
                print(f"Image data size: {len(result['image'])} chars")
                print(f"Used seed: {result.get('seed', 'unknown')}")
                
                # Save the image
                with open("test_output.png", "wb") as f:
                    f.write(base64.b64decode(result['image']))
                print("[SAVED] Saved as test_output.png")
            else:
                print("[FAIL] No image data in response")
        else:
            print(f"[FAIL] Generation failed: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("[TIMEOUT] Generation timed out (>3 minutes)")
    except Exception as e:
        print(f"[FAIL] Generation test failed: {e}")
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    test_api()
