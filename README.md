# 🚀 Qwen-Image RunPod Deployment

Clean and simple deployment of the official Qwen-Image model on RunPod using custom startup commands.

## 📁 Files

- `runpod_startup.sh` - Automated setup script that installs and runs everything
- `RUNPOD_SETUP.md` - Complete step-by-step deployment guide
- `README.md` - This file

## 🎯 Quick Deploy

1. Create RunPod template with PyTorch 2.8.0 base image
2. Add custom startup command pointing to `runpod_startup.sh`  
3. Deploy pod and wait ~10 minutes
4. Access your API at `https://your-pod-8000.proxy.runpod.net`

## ✅ Features

- 🎨 **Official Qwen-Image** from Alibaba's repository
- 🚀 **No authentication** required (simple API)
- 💨 **Fast setup** with automated script  
- 🔤 **Superior text rendering** (English + Chinese)
- ⚡ **Memory optimized** for GPU efficiency

See `RUNPOD_SETUP.md` for detailed instructions!
