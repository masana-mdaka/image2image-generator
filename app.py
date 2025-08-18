import io, os, base64, torch, threading
from fastapi import FastAPI, Response, UploadFile, File, Form
from PIL import Image

app = FastAPI()

MODEL_ID           = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")

# ---- IP-Adapter config (robust defaults) ----
# Where the weights are mounted (we’ll also autodetect a nested sdxl_models/sdxl_models case)
IP_ADAPTER_DIR     = os.getenv("IP_ADAPTER_DIR", "/opt/ipadapter/sdxl_models")
# Optional: force a particular file name; otherwise we’ll pick the first found in the priority list below
IP_ADAPTER_WEIGHT  = os.getenv("IP_ADAPTER_WEIGHT", "")
# FaceID mode (requires different loader args)
IP_ADAPTER_FACEID  = os.getenv("IP_ADAPTER_FACEID", "false").lower() == "true"

# Optional LoRA hook (kept from your original)
LORA_REPO          = os.getenv("LORA_REPO", "")
LORA_WEIGHT_NAME   = os.getenv("LORA_WEIGHT_NAME", "")

dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

_pipe = None
_pipe_lock = threading.Lock()
_warm_started = False

# for /ping diagnostics
_loaded = {
    "model": MODEL_ID,
    "ip_adapter_path": None,
    "ip_adapter_faceid": IP_ADAPTER_FACEID,
    "lora_repo": LORA_REPO or None,
}

@app.on_event("startup")
def _background_warm():
    threading.Thread(target=lambda: get_pipe(), daemon=True).start()

def _find_ip_adapter_file():
    """
    Pick an IP-Adapter SDXL weight file to load.
    - Honors IP_ADAPTER_DIR and IP_ADAPTER_WEIGHT if set.
    - Otherwise tries a priority list of common filenames.
    - Accepts accidental nesting '/sdxl_models/sdxl_models'.
    Returns: (root_dir_for_load, weight_name, is_face_model, image_encoder_dir or None)
    """
    # Resolve base dir and handle accidental nested folder
    base = IP_ADAPTER_DIR
    if not os.path.isdir(base):
        return None, None, False, None
    # If someone ended up with /opt/ipadapter/sdxl_models/sdxl_models, prefer the inner
    nested = os.path.join(base, "sdxl_models")
    if os.path.isdir(nested):
        base = nested

    candidates = []
    if IP_ADAPTER_WEIGHT:
        # Caller wants a specific file name
        candidates = [IP_ADAPTER_WEIGHT]
    else:
        # Common filenames from h94/IP-Adapter for SDXL
        candidates = [
            "ip-adapter-plus_sdxl_vit-h.safetensors",
            "ip-adapter-plus_sdxl_vit-h.bin",
            "ip-adapter_sdxl_vit-h.safetensors",
            "ip-adapter_sdxl_vit-h.bin",
            "ip-adapter_sdxl.safetensors",
            "ip-adapter_sdxl.bin",
            # face variants (will require image_encoder)
            "ip-adapter-plus-face_sdxl_vit-h.safetensors",
            "ip-adapter-plus-face_sdxl_vit-h.bin",
            "ip-adapter-faceid_sdxl.bin",
            "ip-adapter-faceid_sdxl.safetensors",
        ]

    chosen = None
    for name in candidates:
        p = os.path.join(base, name)
        if os.path.isfile(p):
            chosen = name
            break

    if not chosen:
        return None, None, False, None

    is_face = any(k in chosen for k in ["face", "faceid"])
    image_encoder_dir = None
    if is_face:
        # Typical path in h94/IP-Adapter for SDXL FaceID variants
        # Mount your downloaded encoder to: /opt/ipadapter/sdxl_models/image_encoder
        enc = os.path.join(os.path.dirname(base), "image_encoder")
        if os.path.isdir(enc):
            image_encoder_dir = enc

    return base, chosen, is_face, image_encoder_dir

def get_pipe():
    global _pipe, _warm_started
    if _pipe is not None:
        return _pipe
    with _pipe_lock:
        if _pipe is not None:
            return _pipe

        from diffusers import StableDiffusionXLImg2ImgPipeline as SDXLImg2ImgPipeline

        # 1) SDXL base (offline)
        _p = SDXLImg2ImgPipeline.from_pretrained(
            "/opt/model/sdxl",
            torch_dtype=dtype,
            local_files_only=True
        ).to(device)

        try:
            _p.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        # 2) Try to load IP-Adapter (robust)
        try:
            base_dir, weight_name, is_face, image_encoder_dir = _find_ip_adapter_file()
            if base_dir and weight_name:
                # For SDXL, load_ip_adapter can take a folder and weight_name
                # If face model, also pass image_encoder_folder
                if is_face:
                    _p.load_ip_adapter(
                        base_dir, subfolder=None, weight_name=weight_name,
                        image_encoder_folder=image_encoder_dir
                    )
                else:
                    # Non-face SDXL models typically live under a folder we pass as base, no encoder needed
                    _p.load_ip_adapter(
                        base_dir, subfolder=None, weight_name=weight_name
                    )

                _loaded["ip_adapter_path"] = os.path.join(base_dir, weight_name)
            else:
                _loaded["ip_adapter_path"] = None
        except Exception as e:
            # leave adapter unset; pipeline will still run vanilla SDXL
            _loaded["ip_adapter_path"] = f"LOAD_FAILED: {e}"

        # 3) Optional LoRA
        if LORA_REPO:
            kwargs = {"weight_name": LORA_WEIGHT_NAME} if LORA_WEIGHT_NAME else {}
            try:
                _p.load_lora_weights(LORA_REPO, **kwargs)
            except Exception:
                pass

        _pipe = _p
        _warm_started = True
        return _pipe

def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@app.get("/ping")
def ping():
    # quick health, also show what we loaded
    return {"status": "ok", "warm": _warm_started, **_loaded}

@app.post("/invocations")
async def invocations(
    image: UploadFile = File(...),
    prompt: str = Form("cartoon style, clean lines, vibrant colors"),
    negative_prompt: str = Form("low quality, deformed"),
    strength: float = Form(0.55),
    guidance_scale: float = Form(3.5),
    steps: int = Form(20),
    seed: int | None = Form(None),
    use_ip_adapter: bool = Form(True),
    ip_adapter_scale: float = Form(0.8),
    lora_scale: float = Form(0.9),
    return_base64: bool = Form(False),
    out_size: int = Form(512),
):
    out_size = max(256, min(out_size, 768))
    raw = await image.read()
    init = Image.open(io.BytesIO(raw)).convert("RGB")
    pipe = get_pipe()

    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    call_kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init,
        strength=float(strength),
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(steps),
        generator=generator,
        height=out_size,
        width=out_size,
    )

    # Only pass ip_adapter args if we actually loaded one
    if use_ip_adapter and isinstance(_loaded.get("ip_adapter_path"), str) and not _loaded["ip_adapter_path"].startswith("LOAD_FAILED"):
        try:
            call_kwargs["ip_adapter_image"] = init
            pipe.set_ip_adapter_scale(float(ip_adapter_scale))
        except Exception:
            # If adapter wasn’t attached properly, fall back silently
            pass

    if LORA_REPO:
        call_kwargs["cross_attention_kwargs"] = {"scale": float(lora_scale)}

    out_img = pipe(**call_kwargs).images[0]
    png = to_png_bytes(out_img)
    if return_base64:
        return {"image_base64": base64.b64encode(png).decode("utf-8"), "format": "PNG"}
    return Response(content=png, media_type="image/png")
