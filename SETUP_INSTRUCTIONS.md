# Qwen-Image RunPod Setup Instructions

## Quick Start (2x A40 96GB Setup)

### 1. RunPod Configuration

**Container Image:**
```
runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel-ubuntu22.04
```

**Environment Variables:**
```
DEFAULT_MODEL=BF16
HF_HOME=/workspace/.cache/huggingface
```

**Start Command:**
```bash
bash -c "curl -L https://raw.githubusercontent.com/storiesops/qwen-image/refs/heads/main/runpod_startup.sh | sed 's/\r$//' | bash"
```

**Disk Space:** Minimum 80GB volume (100GB+ recommended)

**Exposed Ports:** 8000 (HTTP)

---

## Model Options

Set the `DEFAULT_MODEL` environment variable to choose your model:

### Option 1: BF16 (Recommended for 2x A40 96GB)
```
DEFAULT_MODEL=BF16
```
- **Model:** Original Qwen/Qwen-Image (full precision)
- **Size:** ~41GB
- **VRAM:** ~45GB peak
- **Quality:** Best (100% original quality)
- **Speed:** Fast (no quantization overhead)
- **Recommended for:** Large GPU setups (2x A40, A100, H100)

### Option 2: NUNCHAKU (Recommended for Single GPU 24-48GB)
```
DEFAULT_MODEL=NUNCHAKU
```
- **Model:** nunchaku-tech/nunchaku-qwen-image (INT4 quantized)
- **Size:** ~12GB
- **VRAM:** ~15GB peak
- **Quality:** Very Good (minimal quality loss)
- **Speed:** Fastest (optimized kernels)
- **Recommended for:** L40S (48GB), A40 (48GB), RTX 4090

### Option 3: DF11 (Experimental - Lossless Compression)
```
DEFAULT_MODEL=DF11
```
- **Model:** DFloat11/Qwen-Image-DF11 (lossless compression)
- **Size:** ~28GB
- **VRAM:** ~30GB peak
- **Quality:** Perfect (100% lossless)
- **Speed:** Fast
- **Status:** Requires PyTorch 2.5.1 + CUDA 12.4 (included in setup)
- **Recommended for:** L40S (48GB), A40 (48GB)

---

## API Endpoints

### Synchronous Generation
**POST** `/generate`

```json
{
  "prompt": "a cute dog wearing a red cape, cinematic lighting",
  "negative_prompt": "blurry, low quality, distorted",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 50,
  "true_cfg_scale": 4.0,
  "seed": 42
}
```

**Response:**
```json
{
  "image": "base64_encoded_image_data",
  "seed": 42,
  "success": true
}
```

**Note:** May timeout on Cloudflare proxy (60s limit) for large images. Use async endpoints for production.

---

### Asynchronous Generation (Recommended for Production)

#### 1. Submit Job
**POST** `/generate_async`

```json
{
  "prompt": "a cute dog wearing a red cape, cinematic lighting",
  "negative_prompt": "blurry, low quality, distorted",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 50,
  "true_cfg_scale": 4.0,
  "seed": 42
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

#### 2. Check Status
**GET** `/status/{job_id}`

**Response (processing):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "detail": "Generating image..."
}
```

#### 3. Get Result
**GET** `/result/{job_id}`

**Response (when done):**
```json
{
  "image": "base64_encoded_image_data",
  "seed": 42,
  "success": true
}
```

**Response (still processing):**
- **Status Code:** 202 Accepted
- **Header:** `Retry-After: 10`

---

## Troubleshooting

### Error: "Disk quota exceeded"
**Problem:** Not enough disk space to download model (~41GB for BF16)

**Solution:**
1. Increase RunPod volume size to 100GB+
2. Or switch to smaller model: `DEFAULT_MODEL=NUNCHAKU` (~12GB)

### Error: "CUDA out of memory"
**Problem:** GPU doesn't have enough VRAM

**Solution:**
1. For single GPU <48GB: Use `DEFAULT_MODEL=NUNCHAKU`
2. For BF16, you need at least 48GB VRAM (or 2x GPUs)

### High CPU Usage, Low GPU Usage
**Problem:** Model is loading to CPU instead of GPU

**Check logs for:**
```
INFO:__main__:🔍 Transformer device: cuda:0
INFO:__main__:🔍 VAE device: cuda:0
```

If you see `cpu` instead, check:
1. `DEFAULT_MODEL` is set correctly
2. Disk space is sufficient (the download may have failed)
3. Restart container and check logs from the beginning

### Black Images Generated
**Problem:** Missing `negative_prompt` or `true_cfg_scale=0`

**Solution:** Always include:
```json
{
  "negative_prompt": "blurry, low quality, distorted",
  "true_cfg_scale": 4.0
}
```

---

## Testing

### Test Health Check
```bash
curl https://your-pod-id.proxy.runpod.net/health
```

### Test Sync Generation (Quick Test)
```bash
curl -X POST https://your-pod-id.proxy.runpod.net/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cute dog",
    "negative_prompt": "blurry",
    "width": 512,
    "height": 512,
    "num_inference_steps": 20,
    "true_cfg_scale": 4.0
  }'
```

### Test Async Generation (Production Use)
```bash
# Submit job
JOB_ID=$(curl -X POST https://your-pod-id.proxy.runpod.net/generate_async \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cute dog wearing a red cape, cinematic lighting",
    "negative_prompt": "blurry, low quality, distorted",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "true_cfg_scale": 4.0
  }' | jq -r '.job_id')

echo "Job ID: $JOB_ID"

# Check status (repeat until done)
curl https://your-pod-id.proxy.runpod.net/status/$JOB_ID

# Get result (when status=done)
curl https://your-pod-id.proxy.runpod.net/result/$JOB_ID
```

---

## Performance Notes

### BF16 (2x A40 96GB)
- First generation: ~60-75s (includes model load)
- Subsequent: ~60s @ 1024x1024, 50 steps
- Quality: Best possible
- Stable for long-running production use

### NUNCHAKU (Single GPU)
- First generation: ~20-30s (includes model load)
- Subsequent: ~15-25s @ 1024x1024, 50 steps
- Quality: Excellent (minimal loss vs BF16)
- Most cost-effective option

### DF11 (Experimental)
- Similar speed to BF16
- Requires exactly PyTorch 2.5.1 + CUDA 12.4
- Lossless compression (100% quality)
- May have stability issues on some setups

---

## Changes from Previous Version

1. ✅ **Removed all hardcoded "L40S" and "DFloat11" references** - startup messages now reflect your actual DEFAULT_MODEL choice
2. ✅ **Aggressive disk cleanup** - clears HuggingFace cache before model download to prevent "disk quota exceeded"
3. ✅ **Simplified model loading** - clean if/elif/else structure, no confusing fallbacks
4. ✅ **Default changed to BF16** - recommended for your 2x A40 96GB setup
5. ✅ **Fixed device_map** - uses "auto" for optimal multi-GPU distribution
6. ✅ **Better error messages** - tells you exactly what to check (disk space, GPU memory, env vars)
7. ✅ **Async endpoints working** - no more Cloudflare 524 timeouts

---

## Recommended Setup for Your Use Case

**For 2x A40 (96GB total):**
```
Container: runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel-ubuntu22.04
Volume: 100GB
Environment: DEFAULT_MODEL=BF16
Expected performance: ~60s per 1024x1024 image (50 steps)
```

This gives you the best quality with plenty of VRAM headroom. The model will automatically distribute across both GPUs.
