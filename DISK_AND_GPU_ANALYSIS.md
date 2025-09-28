# 🔍 Qwen-Image Disk Space & GPU Analysis

## 📊 **What We're Actually Downloading**

Our script downloads: **`Qwen/Qwen-Image`** from Hugging Face

### **Model Specifications:**
- **Model Type**: 20B parameter MMDiT (Multimodal Diffusion Transformer)
- **Precision**: BF16 (tries FP8 quantized first, falls back to BF16)
- **Architecture**: Main model + Text encoder components

---

## 💾 **Disk Space Breakdown**

### **Container Disk: 40GB** (Why this size?)
```
🐋 RunPod PyTorch base image:        ~8-10GB
📦 System packages (git, curl):     ~1-2GB  
🐍 PyTorch + CUDA libraries:        ~8-10GB
📚 Python packages:                 ~3-5GB
🔧 Diffusers, Transformers:         ~2-3GB
📁 Temporary files during install:  ~5-10GB
🎯 Safety buffer:                   ~5GB
                                    --------
                                    ~40GB total
```

### **Volume Disk: 80GB** (Why this size?)
```
📱 Qwen-Image Model Files:
  • Full BF16 model:                ~40GB
  • Text encoder:                   ~16GB  
  • Tokenizer & config files:       ~1GB
  • Hugging Face cache overhead:    ~5-10GB

🔄 Runtime Storage:
  • Generated images cache:         ~5GB
  • Model activation cache:         ~5GB
  • Temporary generation files:     ~3GB
  • Log files:                      ~1GB
  • Safety buffer:                  ~5GB
                                    --------
                                    ~80GB total
```

---

## 🎮 **GPU Memory Requirements**

### **What happens in VRAM during inference:**

#### **FP8 Quantized Model (If Available):**
```
🧠 Main model weights:              ~20GB
📝 Text encoder weights:            ~9GB  
⚡ Activation tensors:              ~8-12GB
🎨 VAE decoder:                     ~2-4GB
🔄 Temporary computation buffers:   ~3-5GB
                                    --------
                                    ~42-50GB total
```

#### **BF16 Full Precision Model:**
```
🧠 Main model weights:              ~40GB
📝 Text encoder weights:            ~16GB  
⚡ Activation tensors:              ~12-18GB
🎨 VAE decoder:                     ~3-6GB
🔄 Temporary computation buffers:   ~5-8GB
                                    --------
                                    ~76-88GB total
```

---

## 🖥️ **GPU Recommendations (Updated)**

### **🏆 RTX A6000 (48GB) - $0.79/hr**
```
✅ Can run FP8 quantized model comfortably
✅ All memory optimizations enabled
✅ Good performance with 1024x1024 images
⚠️ May struggle with BF16 full precision
🎯 RECOMMENDED CHOICE
```

### **💪 A100 40GB SXM4 - $1.89/hr** 
```
✅ Reliable for FP8 quantized model
✅ Professional-grade reliability
⚠️ Definitely cannot handle BF16 full precision
💰 More expensive but very stable
```

### **⚠️ RTX 4090 (24GB) - $0.53/hr**
```
❌ Cannot load full Qwen-Image model
❌ Even FP8 quantized (~42GB) won't fit
❌ Would need extreme optimizations/smaller models
🚫 NOT RECOMMENDED for Qwen-Image
```

### **❌ RTX 3080/3090 (10-24GB)**
```
❌ Completely insufficient VRAM
❌ Model won't even load
🚫 WILL NOT WORK
```

---

## ⚡ **Memory Optimizations in Our Script**

Our startup script includes these optimizations:
```python
✅ Attention slicing        # Reduces peak VRAM during attention
✅ VAE slicing             # Processes images in smaller chunks  
✅ VAE tiling              # Further reduces VAE memory usage
✅ CPU offload             # Moves idle model parts to RAM
✅ Memory cache clearing   # Frees unused GPU memory
✅ FP8 quantization        # Tries to load smaller model first
```

---

## 🎯 **Realistic Deployment Scenario**

### **Best Case (RTX A6000 48GB):**
```
1. Script tries FP8 quantized (~29GB model + ~15GB runtime) = ~44GB
2. Fits comfortably with optimizations
3. Good generation speed
4. Reliable operation
```

### **Budget Case (Try smaller model):**
```
1. Modify script to use a smaller diffusion model
2. Or implement advanced chunking/tiling
3. Use 512x512 generation instead of 1024x1024
4. Enable aggressive CPU offloading
```

---

## 🔧 **Script Behavior:**

1. **First attempt**: Load FP8 quantized model (saves ~50% VRAM)
2. **Fallback**: Load BF16 full precision if FP8 not available  
3. **Memory detection**: Enable CPU offload for GPUs < 40GB
4. **All optimizations**: Attention slicing, VAE optimizations, cache clearing

---

## 💡 **Bottom Line:**

- **Disk Space**: 80GB volume needed for model + cache + runtime files
- **Container**: 40GB for system + PyTorch + packages
- **GPU**: RTX A6000 (48GB) minimum for reliable operation
- **Model Size**: ~29GB quantized, ~56GB full precision
- **VRAM Usage**: ~44GB quantized, ~80GB+ full precision

**The original 30GB container + 50GB volume was insufficient!** 🚨
