# 🚀 Qwen-Image on RunPod - Simple Setup Guide

This guide shows you how to deploy Qwen-Image using a **RunPod template** with **custom startup commands** - no Docker building required!

## 🎯 Step 1: Create RunPod Template

1. **Go to RunPod Dashboard** → Templates → New Template

2. **Fill in the template:**
   ```
   Template Name: Qwen-Image-Custom
   Container Image: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
   Container Registry Credentials: (leave empty - public image)
   
   Container Disk: 40 GB
   Volume Disk: 80 GB  
   Volume Mount Path: /workspace
   
   Expose HTTP Ports:
   - Container Port: 8000
   - HTTP Port: 8000
   
   Environment Variables:
   - PYTHONUNBUFFERED=1
   - HF_HOME=/workspace/.cache/huggingface
   ```

3. **Custom Start Command:**
   ```bash
   bash -c "curl -L https://raw.githubusercontent.com/arkodeepsen/qwen-image/main/runpod_startup.sh | bash"
   ```
   
   Or if you want to use the local file:
   ```bash
   bash -c "wget -O /tmp/setup.sh https://your-raw-file-url/runpod_startup.sh && chmod +x /tmp/setup.sh && /tmp/setup.sh"
   ```

## 🖥️ Step 2: Deploy Pod

1. **Create new pod** using your template
2. **Choose GPU (CRITICAL - Model needs lots of VRAM!):**
   - **🏆 Best**: RTX A6000 (48GB) - $0.79/hr - Handles full model perfectly
   - **💪 Good**: A100 40GB SXM4 - $1.89/hr - Reliable for quantized model
   - **⚠️ Risky**: RTX 4090 (24GB) - $0.53/hr - Only works with heavy optimizations
   - **❌ Too Small**: RTX 3080/3090 (10-24GB) - Will likely run out of memory

3. **Deploy and wait** for the startup script to complete (~5-10 minutes)

## 🧪 Step 3: Test Your API

Once your pod is running, you'll get a URL like: `https://abc123-8000.proxy.runpod.net`

### Health Check
```bash
curl https://abc123-8000.proxy.runpod.net/health
```

### Generate Image
```bash
curl -X POST https://abc123-8000.proxy.runpod.net/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful coffee shop with neon sign reading Qwen Coffee",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "seed": 42
  }'
```

### API Documentation
Visit: `https://abc123-8000.proxy.runpod.net/docs`

## 🐍 Step 4: Python Client Example

```python
import requests
import base64
import io
from PIL import Image

# Your RunPod URL
API_URL = "https://abc123-8000.proxy.runpod.net"

def generate_qwen_image(prompt, **kwargs):
    """Generate image using Qwen-Image API"""
    
    payload = {
        "prompt": prompt,
        "width": kwargs.get("width", 1024),
        "height": kwargs.get("height", 1024), 
        "num_inference_steps": kwargs.get("steps", 50),
        "seed": kwargs.get("seed"),
        "negative_prompt": kwargs.get("negative_prompt")
    }
    
    # Remove None values
    payload = {k: v for k, v in payload.items() if v is not None}
    
    try:
        response = requests.post(f"{API_URL}/generate", json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        
        # Decode base64 image
        image_data = base64.b64decode(result["image"])
        image = Image.open(io.BytesIO(image_data))
        
        print(f"✅ Generated image with seed: {result['seed']}")
        return image
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Test generation
    image = generate_qwen_image(
        prompt="A cyberpunk city street with neon signs in Chinese and English",
        width=1024,
        height=1024,
        steps=50,
        seed=12345
    )
    
    if image:
        image.save("qwen_generated.png")
        print("🎨 Image saved as qwen_generated.png")
```

## 🔧 Advantages of This Approach

✅ **No Docker building** - uses official RunPod PyTorch image
✅ **No authentication hassles** - clean, simple API  
✅ **Official Qwen-Image** - directly from Alibaba's repo
✅ **Easy customization** - modify the startup script as needed
✅ **Memory efficient** - automatic attention & VAE slicing
✅ **GPU optimized** - uses latest CUDA & cuDNN

## 🎨 Qwen-Image Specialties

- **🔤 Superior text rendering** (both English & Chinese)
- **🎯 Precise image editing** capabilities  
- **🌍 Multi-language support**
- **📐 Flexible aspect ratios**
- **⚡ Fast generation** with optimizations

## 💡 Tips & Tricks

- **For text rendering**: Be specific about text content in quotes
- **For Chinese text**: The model excels at Chinese characters
- **Memory issues**: Reduce `num_inference_steps` to 20-30 for lower VRAM
- **Higher quality**: Increase `num_inference_steps` to 80-100 for detailed images
- **Consistent results**: Always use the same `seed` value

## 🐛 Troubleshooting

**Pod won't start?**
- Check the logs in RunPod dashboard
- Ensure you have enough VRAM for your chosen GPU

**Generation is slow?**  
- Use fewer inference steps (20-30)
- Reduce image resolution (512x512)

**Out of memory errors?**
- Enable CPU offload by modifying the startup script
- Use a GPU with more VRAM
- Generate smaller images

---

🎉 **That's it!** You now have a clean, simple Qwen-Image API running on RunPod without any authentication headaches!
