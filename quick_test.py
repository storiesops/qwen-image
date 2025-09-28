#!/usr/bin/env python3
"""
Quick inference test with working API
"""
import requests
import json
import time

API_URL = "https://v9edjr0o921q6q-8000.proxy.runpod.net"

def test_generation():
    """Test a simple image generation"""
    print("TESTING QWEN-IMAGE GENERATION")
    print("=" * 40)
    
    # Test prompt
    prompt = "A beautiful sunset over mountains, vibrant colors, photorealistic"
    
    payload = {
        "prompt": prompt,
        "width": 512,
        "height": 512,
        "num_inference_steps": 20
    }
    
    print(f"[*] Prompt: {prompt}")
    print(f"[*] Size: {payload['width']}x{payload['height']}")
    print(f"[*] Steps: {payload['num_inference_steps']}")
    
    try:
        start_time = time.time()
        print("\n[*] Sending generation request...")
        
        response = requests.post(
            f"{API_URL}/generate",
            json=payload,
            timeout=180  # 3 minute timeout
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"[T] Generation time: {generation_time:.1f} seconds")
        print(f"[>] Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("[+] SUCCESS! Image generated")
            
            if "image" in result:
                image_data = result["image"]
                print(f"[I] Base64 image data length: {len(image_data)}")
                
                # Save the image
                import base64
                from datetime import datetime
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"qwen_test_{timestamp}.png"
                
                with open(filename, "wb") as f:
                    f.write(base64.b64decode(image_data))
                    
                print(f"[S] Saved image: {filename}")
                print(f"[+] INFERENCE TEST SUCCESSFUL!")
                
            return True
        else:
            print(f"[-] Error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"[-] Generation failed: {e}")
        return False

if __name__ == "__main__":
    # First check health
    try:
        health = requests.get(f"{API_URL}/health").json()
        print(f"[+] API Health: {health['status']}")
        print(f"[+] GPU: {health['gpu_name']}")
        print(f"[+] VRAM: {health['gpu_memory_used']} / {health['gpu_memory_total']}")
        print()
    except:
        print("[-] Could not get health status")
        
    # Run generation test
    test_generation()
