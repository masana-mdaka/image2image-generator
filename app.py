import io, os, base64, torch, threading
from fastapi import FastAPI, Response, UploadFile, File, Form
from PIL import Image

app = FastAPI()

MODEL_ID          = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
IP_ADAPTER_REPO   = os.getenv("IP_ADAPTER_REPO", "h94/IP-Adapter")
IP_ADAPTER_WEIGHT = os.getenv("IP_ADAPTER_WEIGHT", "ip-adapter_sdxl.bin")
IP_ADAPTER_FACEID = os.getenv("IP_ADAPTER_FACEID", "false").lower() == "true"

LORA_REPO        = os.getenv("LORA_REPO", "")
LORA_WEIGHT_NAME = os.getenv("LORA_WEIGHT_NAME", "")

dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

_pipe = None
_pipe_lock = threading.Lock()
_warm_started = False

@app.on_event("startup")
def _background_warm():
    import threading
    threading.Thread(target=lambda: get_pipe(), daemon=True).start()


def get_pipe():
    global _pipe, _warm_started
    if _pipe is not None:
        return _pipe
    with _pipe_lock:
        if _pipe is not None:
            return _pipe
        from diffusers import StableDiffusionXLImg2ImgPipeline as SDXLImg2ImgPipeline
        # load from baked path, offline
        _pipe = SDXLImg2ImgPipeline.from_pretrained(
            "/opt/model/sdxl",
            torch_dtype=dtype,
            local_files_only=True
        ).to(device)
        try:
            _pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        # Load IP-Adapter from baked path
        try:
            if not IP_ADAPTER_FACEID:
                _pipe.load_ip_adapter(
                    "/opt/ipadapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin"
                )
            else:
                # if you later bake faceid weights, point to that local dir
                _pipe.load_ip_adapter(
                    "h94/IP-Adapter-FaceID", subfolder=None,
                    weight_name="ip-adapter-faceid_sdxl.bin",
                    image_encoder_folder=None
                )
        except Exception:
            pass

        if LORA_REPO:
            kwargs = {"weight_name": LORA_WEIGHT_NAME} if LORA_WEIGHT_NAME else {}
            try:
                _pipe.load_lora_weights(LORA_REPO, **kwargs)
            except Exception:
                pass

        _warm_started = True
        return _pipe

     
 

def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

@app.get("/ping")
def ping():
    # fast health check; do NOT load model here
    return {"status": "ok", "warm": _warm_started}

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
    out_size: int = Form(512),   # ✅ default now 512
):
    # ✅ clamp size
    out_size = max(256, min(out_size, 768))  # safe range 256–768
    raw = await image.read()
    init = Image.open(io.BytesIO(raw)).convert("RGB")
    pipe = get_pipe()

    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    call_kwargs = dict(
        prompt=prompt, negative_prompt=negative_prompt, image=init,
        strength=float(strength), guidance_scale=float(guidance_scale),
        num_inference_steps=int(steps), generator=generator,
        height=out_size, width=out_size,
    )

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

