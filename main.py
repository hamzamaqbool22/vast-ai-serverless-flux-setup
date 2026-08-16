import base64
import io
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

from PIL import Image
import torch
from diffusers import Flux2KleinPipeline


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s: %(message)s",
)

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)

logger = logging.getLogger("api")


# ============================================================
# App
# ============================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

torch.backends.cudnn.benchmark = True


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "/workspace/flux"

device = "cuda" if torch.cuda.is_available() else "cpu"

executor = ThreadPoolExecutor(max_workers=1)


# ============================================================
# Request model for Serverless
# ============================================================


class GenerateRequest(BaseModel):
    image: str
    prompt: str
    height: int = 768
    width: int = 1024


# ============================================================
# Load model
# ============================================================

logger.warning("Loading FLUX.2-klein-4B...")

pipe = Flux2KleinPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
)

pipe.to(device)

logger.warning(f"Model moved to {device}")


# ============================================================
# Warmup
# ============================================================

if device == "cuda":
    try:
        logger.warning("Warmup run...")

        dummy = Image.new(
            "RGB",
            (512, 512),
            "white",
        )

        with torch.inference_mode():
            pipe(
                image=dummy,
                prompt="test",
                num_inference_steps=1,
                height=512,
                width=512,
            )

        torch.cuda.synchronize()

        logger.warning("Warmup done")

    except Exception:
        logger.exception("Warmup failed")
        raise


# IMPORTANT:
# Vast uses this log message to know the worker is ready to
# start benchmarking.
logger.warning("VAST_MODEL_READY")


# ============================================================
# Inference
# ============================================================


def run_inference(
    image: Image.Image,
    prompt: str,
    h: int,
    w: int,
):
    start = time.time()

    # Limit huge input images.
    image.thumbnail((1600, 1600))

    with torch.inference_mode():
        result = pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=8,
            guidance_scale=1.5,
            height=h,
            width=w,
        ).images[0]

    buf = io.BytesIO()

    result.save(
        buf,
        format="PNG",
        optimize=True,
    )

    buf.seek(0)

    elapsed = time.time() - start

    logger.warning(f"Inference completed in {elapsed:.2f}s")

    return buf


# ============================================================
# Helpers
# ============================================================


def decode_base64_image(value: str) -> Image.Image:
    try:
        # Supports both:
        # data:image/png;base64,...
        # and plain base64
        if "," in value and value.startswith("data:"):
            value = value.split(",", 1)[1]

        raw = base64.b64decode(value)

        image = Image.open(io.BytesIO(raw)).convert("RGB")

        return image

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid base64 image",
        ) from exc


def encode_image_base64(buf: io.BytesIO) -> str:
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================
# Basic routes
# ============================================================


@app.get("/")
def root():
    return {"status": "api is running"}


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": device,
    }


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


# ============================================================
# Existing normal API
# ============================================================


@app.post("/edit")
async def edit(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    height: Optional[int] = Form(768),
    width: Optional[int] = Form(1024),
):
    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Missing content type",
        )

    if not file.content_type.startswith("image"):
        raise HTTPException(
            status_code=400,
            detail="Invalid image",
        )

    data = await file.read()

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Bad image",
        ) from exc

    loop = asyncio.get_running_loop()

    buf = await loop.run_in_executor(
        executor,
        run_inference,
        image,
        prompt,
        height,
        width,
    )

    return StreamingResponse(
        buf,
        media_type="image/png",
    )


# ============================================================
# Vast Serverless API
# ============================================================


@app.post("/generate")
async def generate(request: GenerateRequest):
    image = decode_base64_image(request.image)

    loop = asyncio.get_running_loop()

    buf = await loop.run_in_executor(
        executor,
        run_inference,
        image,
        request.prompt,
        request.height,
        request.width,
    )

    image_base64 = encode_image_base64(buf)

    return {
        "success": True,
        "image": image_base64,
        "format": "png",
        "width": request.width,
        "height": request.height,
    }
