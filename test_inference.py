#!/usr/bin/env python3
"""
Test inference with the running Qwen-Image API
"""
import requests
import json
import time
from datetime import datetime

# Your API endpoint
API_URL = "https://v9edjr0o921q6q-8000.proxy.runpod.net"

def test_api_health():
    """Test if API is responding"""
    print("[*] Testing API health...")
    try:
        response = requests.get(f"{API_URL}/")
        print(f"[+] API Status: {response.status_code}")
        print(f"[>] Response: {response.json()}")
        return True
    except Exception as e:
        print(f"[-] API Health Check Failed: {e}")
        return False

def get_model_info():
    """Get model information"""
    print("\n[*] Getting model info...")
    try:
        response = requests.get(f"{API_URL}/models/info")
        print(f"[+] Model Info: {response.status_code}")
        info = response.json()
        for key, value in info.items():
            print(f"   {key}: {value}")
        return True
    except Exception as e:
        print(f"[-] Model Info Failed: {e}")
        return False

def generate_image(prompt, width=512, height=512, num_inference_steps=20):
    """Generate an image"""
    print(f"\n[*] Generating image: '{prompt}'")
    
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps
    }
    
    print(f"[>] Parameters: {json.dumps(payload, indent=2)}")
    
    try:
        start_time = time.time()
        print("[*] Sending request to API...")
        
        response = requests.post(
            f"{API_URL}/generate",
            json=payload,
            timeout=300  # 5 minute timeout for generation
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"[+] Response Status: {response.status_code}")
        print(f"[T] Generation Time: {generation_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"[+] Success! Image generated")
            if "image" in result:
                print(f"[I] Image data length: {len(result['image'])} characters (base64)")
                
                # Save image if base64 data is returned
                if result['image']:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"qwen_generated_{timestamp}.png"
                    
                    import base64
                    with open(filename, "wb") as f:
                        f.write(base64.b64decode(result['image']))
                    print(f"[S] Saved image as: {filename}")
            
            return True
        else:
            print(f"[-] Error: {response.status_code}")
            print(f"[E] Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("[T] Request timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"[-] Generation Failed: {e}")
        return False

def main():
    """Main test function"""
    print("QWEN-IMAGE API INFERENCE TEST")
    print("=" * 50)
    
    # Test 1: API Health
    if not test_api_health():
        print("[-] API not responding, stopping tests")
        return
    
    # Test 2: Model Info
    get_model_info()
    
    # Test 3: Simple image generation
    test_prompts = [
        "A beautiful sunset over mountains with vibrant orange and purple clouds",
        "A cute robot cat sitting in a garden with flowers",
        "Abstract digital art with flowing colors and geometric patterns"
    ]
    
    print(f"\n[*] Testing {len(test_prompts)} image generations...")
    
    success_count = 0
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*20} TEST {i}/{len(test_prompts)} {'='*20}")
        if generate_image(prompt):
            success_count += 1
        print("-" * 60)
    
    print(f"\n[R] FINAL RESULTS:")
    print(f"[+] Successful generations: {success_count}/{len(test_prompts)}")
    print(f"[%] Success rate: {success_count/len(test_prompts)*100:.1f}%")

if __name__ == "__main__":
    main()
