#!/usr/bin/env python3
"""
Test API docs and generate an image
"""
import requests
import json
import base64
import time
from datetime import datetime

API_URL = "https://v9edjr0o921q6q-8000.proxy.runpod.net"

def check_docs():
    """Check what's available in the API docs"""
    print("CHECKING API DOCUMENTATION")
    print("=" * 40)
    
    docs_url = f"{API_URL}/docs"
    print(f"[*] API Docs URL: {docs_url}")
    print("[*] Opening docs in browser would show FastAPI Swagger UI")
    
    # Let's try to get the OpenAPI spec
    try:
        openapi_response = requests.get(f"{API_URL}/openapi.json")
        if openapi_response.status_code == 200:
            spec = openapi_response.json()
            print(f"[+] OpenAPI Spec retrieved")
            print(f"[>] API Title: {spec.get('info', {}).get('title', 'Unknown')}")
            print(f"[>] API Version: {spec.get('info', {}).get('version', 'Unknown')}")
            
            # Show available endpoints
            if 'paths' in spec:
                print("\n[*] Available Endpoints:")
                for path, methods in spec['paths'].items():
                    for method in methods.keys():
                        print(f"   {method.upper()} {path}")
            
            return spec
        else:
            print(f"[-] Could not get OpenAPI spec: {openapi_response.status_code}")
            return None
    except Exception as e:
        print(f"[-] Error getting API spec: {e}")
        return None

def test_health():
    """Test health endpoint"""
    print("\nTESTING HEALTH ENDPOINT")
    print("-" * 30)
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print("[+] Health Status:")
            for key, value in health.items():
                print(f"   {key}: {value}")
            return health
        else:
            print(f"[-] Health check failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"[-] Health check error: {e}")
        return None

def generate_image():
    """Generate a test image"""
    print("\nTESTING IMAGE GENERATION")
    print("-" * 30)
    
    # Test prompt
    prompt = "A serene mountain landscape at sunset, with vibrant orange and purple clouds reflecting on a calm lake, photorealistic, detailed"
    
    payload = {
        "prompt": prompt,
        "width": 768,  # Try a slightly larger size
        "height": 768,
        "num_inference_steps": 25,  # A few more steps for quality
        "guidance_scale": 7.5  # Add guidance scale if supported
    }
    
    print(f"[*] Prompt: {prompt}")
    print(f"[*] Parameters:")
    for key, value in payload.items():
        print(f"   {key}: {value}")
    
    try:
        start_time = time.time()
        print(f"\n[*] Sending generation request to {API_URL}/generate...")
        
        response = requests.post(
            f"{API_URL}/generate",
            json=payload,
            timeout=300  # 5 minute timeout for generation
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"[T] Generation took: {generation_time:.1f} seconds")
        print(f"[>] HTTP Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("[+] SUCCESS! Image generated")
            
            # Check what we got back
            print(f"[>] Response keys: {list(result.keys())}")
            
            if "image" in result and result["image"]:
                image_data = result["image"]
                print(f"[I] Base64 image data length: {len(image_data)} characters")
                
                # Save the image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"qwen_generated_{timestamp}.png"
                
                try:
                    # Decode and save
                    image_bytes = base64.b64decode(image_data)
                    with open(filename, "wb") as f:
                        f.write(image_bytes)
                    
                    print(f"[S] Image saved as: {filename}")
                    print(f"[S] File size: {len(image_bytes)} bytes")
                    print(f"[+] GENERATION SUCCESSFUL!")
                    
                    return True
                    
                except Exception as save_error:
                    print(f"[-] Error saving image: {save_error}")
                    return False
            else:
                print("[-] No image data in response")
                print(f"[>] Response: {result}")
                return False
                
        elif response.status_code == 422:
            print("[-] Validation error (422) - checking response...")
            try:
                error_detail = response.json()
                print(f"[E] Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"[E] Raw response: {response.text}")
            return False
        else:
            print(f"[-] Generation failed with status {response.status_code}")
            print(f"[E] Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("[-] Request timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"[-] Generation error: {e}")
        return False

def main():
    """Main test function"""
    print("QWEN-IMAGE API TESTING")
    print("=" * 50)
    
    # 1. Check API documentation
    spec = check_docs()
    
    # 2. Test health endpoint
    health = test_health()
    if not health:
        print("[-] API not healthy, stopping tests")
        return
    
    # 3. Test image generation
    success = generate_image()
    
    print(f"\n{'='*50}")
    if success:
        print("[+] ALL TESTS PASSED! Your Qwen-Image API is working perfectly!")
    else:
        print("[-] Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
