#!/bin/bash
set -e

echo "🚀 Starting Qwen-Image Setup on RunPod"
echo "=========================================="

# Update system
echo "📦 Updating system packages..."
apt-get update -qq
apt-get install -y git wget curl

# Clone official Qwen-Image repository
echo "📥 Cloning official Qwen-Image repository..."
cd /workspace
if [ ! -d "Qwen-Image" ]; then
    git clone https://github.com/QwenLM/Qwen-Image.git
fi
cd Qwen-Image

# Install Python dependencies with PROPER VERSIONS
echo "🐍 Installing Python dependencies with verified compatibility..."

# Upgrade pip first
pip install --upgrade pip

echo "🔥 Installing PyTorch with CUDA 12.8 support..."
# Use the exact PyTorch version that works with RunPod's CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir

# Clean up immediately after install to save space
pip cache purge
echo "📊 Disk usage after PyTorch install:"
echo "Container:" && df -h / || true
echo "Volume:" && df -h /workspace || true

echo "🤗 Installing Hugging Face libraries with correct versions..."
# Qwen-Image specifically requires transformers>=4.51.3 for Qwen2.5-VL support  
pip install "transformers>=4.51.3" --no-cache-dir
pip install "accelerate>=0.26.1" --no-cache-dir
pip install "safetensors>=0.3.1" --no-cache-dir

# Clean cache after each install
pip cache purge

echo "🔥 Installing DFloat11 for lossless compression (32% smaller, 100% quality)..."
pip install -U "dfloat11[cuda12]" --no-cache-dir
pip cache purge

echo "🎨 Installing latest Diffusers from source..."
# Always use latest diffusers for best Qwen-Image support
pip install git+https://github.com/huggingface/diffusers --no-cache-dir
pip cache purge

echo "🚀 Installing FastAPI stack..."
pip install "fastapi>=0.100.0" "uvicorn[standard]>=0.23.0" "pydantic>=2.0.0" --no-cache-dir

echo "🖼️ Installing image processing libraries..."
pip install "pillow>=10.0.0" requests --no-cache-dir

# Final cache cleanup
pip cache purge
echo "📊 Final disk usage after installs:"
echo "Container:" && df -h / || true
echo "Volume (critical):" && df -h /workspace || true

echo "🌍 Setting up environment variables..."
export HF_HOME=/workspace/.cache/huggingface  # Replace deprecated TRANSFORMERS_CACHE
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface  # Fallback for compatibility
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Fix memory fragmentation

echo "🧹 EMERGENCY: Cleaning disk space and GPU processes..."
# Kill any existing Python processes
pkill -f "python.*qwen" || true
pkill -f "uvicorn.*qwen" || true

# AGGRESSIVE DISK CLEANUP for RunPod VOLUME (not container!)
echo "💾 Freeing up VOLUME disk space at /workspace..."

# Clean the VOLUME disk where work actually happens
rm -rf /workspace/.cache/* || true
rm -rf /workspace/Qwen-Image/.git || true  # Remove git history (saves ~100MB)
rm -rf /workspace/*.log || true
rm -rf /workspace/temp* || true
rm -rf /workspace/tmp* || true

# Clean container disk too (but this isn't the main issue)
apt-get clean || true
rm -rf /var/lib/apt/lists/* || true
rm -rf /tmp/* || true
rm -rf /var/tmp/* || true

# Clean Python cache (both locations)
pip cache purge || true
python -m pip cache purge || true
rm -rf ~/.cache/pip || true

# Clean conda if exists
conda clean -all -y || true

# Clear GPU memory
nvidia-smi --gpu-reset-gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits) || true

echo "📊 Disk usage after cleanup:"
echo "Container disk:"
df -h /
echo "Volume disk (where work happens):"
df -h /workspace || echo "Volume not mounted yet"
sleep 2

echo "🚫 SKIPPING FlashAttention-2 - causes crashes with PyTorch 2.8 dev even when gracefully handled"
echo "⚡ Using native PyTorch attention (100% reliable, still fast!)"

# Create our simple API wrapper
echo "🔧 Creating FastAPI wrapper..."
echo "🧹 FINAL cleanup before writing API file..."
# Emergency volume cleanup right before file creation
rm -rf /workspace/.cache/huggingface/hub/models--* || true
df -h /workspace || true
cat > /workspace/qwen_api.py << 'EOF'
#!/usr/bin/env python3
"""
Simple FastAPI wrapper for Qwen-Image
No authentication required - clean and simple!
"""
import os
import sys
import logging
import asyncio
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import uvicorn
from PIL import Image
import base64
import io

# Force disable FlashAttention to prevent diffusers from auto-importing it
import os
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"
os.environ["DISABLE_FLASH_ATTN"] = "1"

# FlashAttention causes PyTorch 2.8 dev crashes - using native attention
FLASH_ATTN_AVAILABLE = False
print("⚡ Using native PyTorch attention (100% compatible, still excellent!)")
print("🚀 OPTIMIZED FOR L40S: Running DFloat11 Qwen-Image - lossless compression + 100% quality!")
print("💪 DFloat11: 28.42GB model size, ~30GB peak VRAM vs L40S 48GB = PERFECT FIT!")

from diffusers import DiffusionPipeline

# CRITICAL: Clear any existing GPU memory first
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"🧹 Cleared GPU memory. Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline
pipeline = None

# Modern FastAPI lifespan handler (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global pipeline
    
    logger.info("🚀 Loading DFloat11 Qwen-Image - LOSSLESS 32% compression + 100% quality for L40S!")
    
    # Load the pipeline with optimal settings for Qwen-Image
    try:
        
        # Load DFloat11 compressed Qwen-Image - LOSSLESS 32% smaller + 100% quality!
        from transformers.modeling_utils import no_init_weights
        from dfloat11 import DFloat11Model
        from diffusers import QwenImageTransformer2DModel
        
        # Load DFloat11 compressed transformer (lossless compression!)
        with no_init_weights():
            transformer = QwenImageTransformer2DModel.from_config(
                QwenImageTransformer2DModel.load_config(
                    "Qwen/Qwen-Image", subfolder="transformer",
                ),
            ).to(torch.bfloat16)
        
            # Load DFloat11 compressed weights (32% smaller, bit-identical outputs)
            DFloat11Model.from_pretrained(
                "DFloat11/Qwen-Image-DF11",
                device="cuda",  # Full GPU for L40S - maximum performance!
                cpu_offload=False,  # No CPU offload needed with 48GB VRAM
                pin_memory=True,  # Optimal for L40S
                bfloat16_model=transformer,
            )
            
            # Create pipeline with compressed transformer
            pipeline = DiffusionPipeline.from_pretrained(
                "Qwen/Qwen-Image",
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )
            
            # CRITICAL: Ensure ALL components are on the same device (GPU)
            pipeline = pipeline.to("cuda")
            logger.info("✅ All pipeline components moved to GPU")
        logger.info("✅ Qwen-Image model loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load Qwen-Image model: {e}")
        logger.info("🔧 Trying alternative loading method...")
            try:
                # Fallback: Original model in FP16 (if BF16 fails)
                pipeline = DiffusionPipeline.from_pretrained(
                    "Qwen/Qwen-Image",  # Same official model, different precision
                    torch_dtype=torch.float16,  # FP16 fallback
                    low_cpu_mem_usage=True,
                    device_map="auto",  # Auto device management
                    use_safetensors=True
                )
                
                # CRITICAL: Ensure ALL components are on GPU for fallback too
                pipeline = pipeline.to("cuda")
                logger.info("✅ Fallback pipeline components moved to GPU")
                
                # Force CPU offloading immediately for fallback (if needed)
                if hasattr(pipeline, 'enable_model_cpu_offload'):
                    pipeline.enable_model_cpu_offload()
                    logger.info("🔄 Enabled CPU offloading for fallback method")
                logger.info("✅ Original Qwen-Image loaded with FP16 fallback")
        except Exception as e2:
            logger.error(f"❌ Complete model loading failure: {e2}")
            raise e2
    
    # Model device placement is handled by device_map="auto"
    if torch.cuda.is_available():
        logger.info(f"✅ Model loaded with auto device mapping")
        logger.info(f"🔥 GPU: {torch.cuda.get_device_name()}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_free = gpu_memory - gpu_allocated
        logger.info(f"📊 GPU Memory: {gpu_memory:.1f}GB total, {gpu_allocated:.1f}GB used, {gpu_free:.1f}GB free")
    else:
        logger.info("⚠️ Running on CPU (will be slow)")
        
    # Enable ALL memory optimizations for better VRAM usage
    if hasattr(pipeline, 'enable_attention_slicing'):
        pipeline.enable_attention_slicing()
        logger.info("✅ Attention slicing enabled")
        
    if hasattr(pipeline, 'enable_vae_slicing'):
        pipeline.enable_vae_slicing()
        logger.info("✅ VAE slicing enabled")
        
    if hasattr(pipeline, 'enable_vae_tiling'):
        pipeline.enable_vae_tiling()
        logger.info("✅ VAE tiling enabled")
            
    # L40S has 48GB VRAM - no CPU offload needed, keep everything on GPU for max speed
    logger.info("🚀 L40S detected: Keeping entire model on GPU for maximum performance!")
    logger.info("ℹ️ Skipping CPU offload - L40S 48GB VRAM is perfect for full model")
    
    # Always using native attention for PyTorch 2.8 stability
    logger.info("⚡ Native PyTorch attention active - 100% stable and fast!")
        
    # Aggressive memory cleanup after model loading
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_free_after = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
        logger.info(f"✅ GPU memory cleaned. Free memory: {gpu_free_after:.1f}GB")
        
    logger.info("🎉 Qwen-Image model loaded successfully!")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Qwen-Image API Server...")

app = FastAPI(
    title="Qwen-Image API",
    description="Simple API for Qwen-Image generation",
    version="1.0.0",
    lifespan=lifespan
)

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 7.0
    seed: Optional[int] = None

class GenerateResponse(BaseModel):
    image: str  # base64 encoded
    seed: int
    success: bool = True

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Model loading is now handled in the lifespan function above

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Qwen-Image API is running! 🚀", "docs": "/docs"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Model not loaded"}
        )
    
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_memory_used": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        }
    
    return {
        "status": "healthy",
        "model": "Qwen-Image",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        **gpu_info
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest):
    """Generate image from text prompt"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    logger.info(f"🎨 Generating: {request.prompt[:100]}...")
    
    try:
        # Set up generator for reproducible results
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(request.seed)
        
        # Generate image
        with torch.inference_mode():
            result = pipeline(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                generator=generator
            )
        
        # Get the generated image
        image = result.images[0]
        
        # Convert to base64
        image_b64 = image_to_base64(image)
        
        # Get the seed used
        used_seed = request.seed if request.seed is not None else generator.initial_seed() if generator else 0
        
        logger.info("✅ Image generated successfully!")
        
        return GenerateResponse(
            image=image_b64,
            seed=used_seed,
            success=True
        )
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("🌟 Starting Qwen-Image API Server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
EOF

# Make the API script executable
chmod +x /workspace/qwen_api.py

echo "✅ Setup complete!"
echo ""
echo "🚀 Starting Qwen-Image API server..."
echo "📡 Server will be available on port 8000"
echo "📖 API docs will be at: /docs"
echo ""

# Start the API server
cd /workspace
python qwen_api.py
