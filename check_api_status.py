#!/usr/bin/env python3
"""
Simple API status checker for Qwen-Image API
"""
import requests
import time

API_URL = "https://v9edjr0o921q6q-8000.proxy.runpod.net"

def check_status():
    """Check API status with detailed response"""
    print(f"[*] Checking API at: {API_URL}")
    
    try:
        response = requests.get(f"{API_URL}/", timeout=10)
        print(f"[>] Status Code: {response.status_code}")
        print(f"[>] Headers: {dict(response.headers)}")
        print(f"[>] Content: {response.text[:500]}...")
        
        if response.status_code == 200:
            print("[+] API is ready!")
            return True
        elif response.status_code == 503:
            print("[*] Service unavailable - model likely still loading...")
            return False
        else:
            print(f"[-] Unexpected status: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("[-] Request timed out")
        return False
    except Exception as e:
        print(f"[-] Error: {e}")
        return False

def wait_for_ready(max_wait=300):
    """Wait for API to become ready"""
    print(f"[*] Waiting for API to become ready (max {max_wait} seconds)...")
    
    start_time = time.time()
    attempt = 0
    
    while time.time() - start_time < max_wait:
        attempt += 1
        print(f"\n--- Attempt {attempt} ---")
        
        if check_status():
            elapsed = time.time() - start_time
            print(f"[+] API ready after {elapsed:.1f} seconds!")
            return True
        
        print("[*] Waiting 15 seconds before next check...")
        time.sleep(15)
    
    print(f"[-] API not ready after {max_wait} seconds")
    return False

if __name__ == "__main__":
    print("QWEN-IMAGE API STATUS CHECKER")
    print("=" * 40)
    
    # Check current status
    if check_status():
        print("\n[+] API is ready! You can run inference tests.")
    else:
        print("\n[*] API not ready yet. Monitoring...")
        if wait_for_ready():
            print("\n[+] API is now ready! You can run inference tests.")
        else:
            print("\n[-] API failed to become ready. Check RunPod logs.")
