FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Basics
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# OS deps for PIL/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl ca-certificates git libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python deps (make sure requirements.txt includes: fastapi, uvicorn[standard], pillow, diffusers==0.30.2, transformers==4.43.3, accelerate, safetensors, python-multipart, boto3, etc.)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- Pin Torch/TorchVision (CUDA 12.1) and ensure NMS exists
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    --force-reinstall 'torch==2.4.1' 'torchvision==0.19.1' 'torchaudio==2.4.1' && \
    pip uninstall -y xformers || true

# quick sanity check (build will fail fast if wrong)
RUN python - <<'PY'
import torch, torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
from torchvision.ops import nms
print("nms OK")
import torch.backends.cuda as cb
print("flash_attn_flag:", hasattr(cb, "is_flash_attention_available"))
PY

# ---- Model caches live on SageMaker's writable volume so downloads persist
ENV HF_HOME=/opt/ml/model/hf-cache \
    TRANSFORMERS_CACHE=/opt/ml/model/hf-cache \
    DIFFUSERS_CACHE=/opt/ml/model/hf-cache \
    HUGGINGFACE_HUB_CACHE=/opt/ml/model/hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=0

# App
COPY app.py .

# Network
EXPOSE 8080

# Uvicorn app server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
