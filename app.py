import io, os, base64, torch
from fastapi import FastAPI, Response, UploadFile, File, Form
from PIL import Image
# Use the concrete SDXL img2img pipeline (avoid auto-pipeline importing extra optional deps)
from diffusers import StableDiffusionXLImg2ImgPipeline as SDXLImg2ImgPipeline

app = FastAPI()

# -------- Config (env) --------
MODEL_ID          = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
IP_ADAPTER_REPO   = os.getenv("IP_ADAPTER_REPO", "h94/IP-Adapter")         # default: Plus for SDXL
IP_ADAPTER_WEIGHT = os.getenv("IP_ADAPTER_WEIGHT", "ip-adapter_sdxl.bin")
IP_ADAPTER_FACEID = os.getenv("IP_ADAPTER_FACEID", "false").lower() == "true"

LORA_REPO         = os.getenv("LORA_REPO", "")         # e.g. "ntc-ai/SDXL-LoRA-slider.cartoon"
LORA_WEIGHT_NAME  = os.getenv("LORA_WEIGHT_NAME", "")  # e.g. "cartoon.safetensors"

dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Pipeline load --------
pipe = SDXLImg2ImgPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
pipe = pipe.to(device)
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pass

# IP-Adapter (identity/style guidance) — optional
try:
    if not IP_ADAPTER_FACEID:
        pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder="sdxl_models", weight_name=IP_ADAPTER_WEIGHT)
    else:
        pipe.load_ip_adapter(
            "h94/IP-Adapter-FaceID",
            subfolder=None,
            weight_name="ip-adapter-faceid_sdxl.bin",
            image_encoder_folder=None
        )
except Exception:
    # OK to proceed without IP-Adapter if weights not available
    pass

# Optional LoRA style
if LORA_REPO:
    try:
        kwargs = {"weight_name": LORA_WEIGHT_NAME} if LORA_WEIGHT_NAME else {}
        pipe.load_lora_weights(LORA_REPO, **kwargs)
    except Exception:
        pass

def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/invocations")
async def invocations(
    image: UploadFile = File(...),
    prompt: str = Form("cartoon style, clean lines, vibrant colors"),
    negative_prompt: str = Form("low quality, deformed"),
    strength: float = Form(0.55),
    guidance_scale: float = Form(3.5),
    steps: int = Form(20),
    seed: int | None = Form(None),
    # Adapters
    use_ip_adapter: bool = Form(True),
    ip_adapter_scale: float = Form(0.8),
    lora_scale: float = Form(0.9),
    # Output
    return_base64: bool = Form(False),
    out_size: int = Form(1024),
):
    raw = await image.read()
    init = Image.open(io.BytesIO(raw)).convert("RGB")

    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    call_kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init,
        strength=float(strength),
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(steps),
        generator=generator,
        height=int(out_size),
        width=int(out_size),
    )

    # Apply adapters if present
    if use_ip_adapter:
        try:
            call_kwargs["ip_adapter_image"] = init
            pipe.set_ip_adapter_scale(float(ip_adapter_scale))
        except Exception:
            pass

    if LORA_REPO:
        call_kwargs["cross_attention_kwargs"] = {"scale": float(lora_scale)}

    out_img = pipe(**call_kwargs).images[0]
    png = to_png_bytes(out_img)

    if return_base64:
        return {"image_base64": base64.b64encode(png).decode("utf-8"), "format": "PNG"}
    return Response(content=png, media_type="image/png")
