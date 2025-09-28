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

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.35.0 diffusers>=0.24.0 accelerate>=0.24.0
pip install fastapi uvicorn[standard] pydantic requests pillow
pip install git+https://github.com/huggingface/diffusers
pip install xformers --no-deps

# Create our simple API wrapper
echo "🔧 Creating FastAPI wrapper..."
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
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import uvicorn
from PIL import Image
import base64
import io
from diffusers import DiffusionPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Qwen-Image API",
    description="Simple API for Qwen-Image generation",
    version="1.0.0"
)

# Global pipeline
pipeline = None

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

@app.on_event("startup")
async def load_model():
    """Load the Qwen-Image model"""
    global pipeline
    
    logger.info("🚀 Loading Qwen-Image model...")
    
    try:
        # Load the pipeline with quantization for better memory efficiency
        try:
            # Try to load FP8 quantized version first (saves ~50% VRAM)
            logger.info("🔥 Attempting to load FP8 quantized model...")
            pipeline = DiffusionPipeline.from_pretrained(
                "Qwen/Qwen-Image",
                torch_dtype=torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.bfloat16,
                use_safetensors=True,
                variant="fp8"
            )
            logger.info("✅ FP8 quantized model loaded successfully!")
        except Exception as e:
            logger.warning(f"⚠️ FP8 model not available, falling back to BF16: {e}")
            # Fallback to standard BF16 model
            pipeline = DiffusionPipeline.from_pretrained(
                "Qwen/Qwen-Image", 
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            )
            logger.info("✅ BF16 model loaded (requires more VRAM)")
        
        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")
            logger.info(f"✅ Model loaded on GPU: {torch.cuda.get_device_name()}")
            logger.info(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
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
            
        if hasattr(pipeline, 'enable_model_cpu_offload'):
            # Only enable CPU offload if we detect limited VRAM
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
            if gpu_memory < 40:  # Less than 40GB VRAM
                pipeline.enable_model_cpu_offload()
                logger.info("✅ Model CPU offload enabled (limited VRAM detected)")
            else:
                logger.info("ℹ️ CPU offload skipped (sufficient VRAM available)")
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✅ GPU memory cache cleared")
            
        logger.info("🎉 Qwen-Image model loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        pipeline = None

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
