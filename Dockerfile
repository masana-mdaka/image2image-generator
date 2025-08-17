FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# System deps (image libs for PIL/OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl ca-certificates git libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Torch 2.4.1 adds cuda.is_flash_attention_available; pair with cu121 wheels
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    --force-reinstall 'torch==2.4.1' 'torchvision==0.19.1' 'torchaudio==2.4.1'

# make sure xformers is not present (it’s optional and causing import errors)
RUN pip uninstall -y xformers || true


# Sanity check: NMS + flash-attention flag
RUN python - <<'PY'
import torch, torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
from torchvision.ops import nms
print("nms OK")
import torch.backends.cuda as cb
print("flash_attn_flag:", hasattr(cb, "is_flash_attention_available"))
PY

# ---- local HF caches (no net at runtime)
ENV HF_HOME=/opt/hf-cache \
    TRANSFORMERS_CACHE=/opt/hf-cache \
    HUGGINGFACE_HUB_CACHE=/opt/hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=0

# ---- pre-download SDXL base and IP-Adapter into the image
RUN python - <<'PY'
from diffusers import StableDiffusionXLImg2ImgPipeline as P
from huggingface_hub import snapshot_download
import os, shutil

os.makedirs("/opt/model", exist_ok=True)
os.makedirs("/opt/ipadapter", exist_ok=True)

# 1) SDXL base pipeline (weights+configs) → /opt/model/sdxl
pipe = P.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe.save_pretrained("/opt/model/sdxl")
del pipe

# 2) IP-Adapter SDXL bin → /opt/ipadapter/sdxl_models/ip-adapter_sdxl.bin
snapshot_download(
    repo_id="h94/IP-Adapter",
    allow_patterns=["sdxl_models/ip-adapter_sdxl.bin"],
    local_dir="/opt/ipadapter",
    local_dir_use_symlinks=False
)
PY

# readability at runtime
RUN chmod -R a+rX /opt/model /opt/ipadapter /opt/hf-cache



# 2) Copy code last (fast rebuilds on code changes)
COPY app.py .

# Persist HF cache (so model weights survive container restarts)
VOLUME ["/opt/hf-cache"]

# SageMaker expects 8080 and /ping + /invocations
EXPOSE 8080
COPY entrypoint.sh /usr/local/bin/entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint"]
