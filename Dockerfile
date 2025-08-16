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

# 2) Copy code last (fast rebuilds on code changes)
COPY app.py .

# Persist HF cache (so model weights survive container restarts)
VOLUME ["/opt/hf-cache"]

# SageMaker expects 8080 and /ping + /invocations
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
