import base64
import io
import os
import time
import traceback
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, JSONResponse
from pydantic import BaseModel, Field, field_validator

from PIL import Image
import torch


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
# Configuration
# ============================================================

MODEL_PATH = "/models/flux"
API_HOST = "127.0.0.1"
API_PORT = 18000

MIN_DIM = 64
MAX_DIM = 1024
DIM_MULTIPLE = 16

NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"

executor = ThreadPoolExecutor(max_workers=1)

pipe = None
model_ready = False

torch.backends.cudnn.benchmark = True

print(
    f"FLUX API importing cwd={os.getcwd()} model={MODEL_PATH} "
    f"cuda={torch.cuda.is_available()} device={device}",
    flush=True,
)


# ============================================================
# Request model for Serverless
# ============================================================


class GenerateRequest(BaseModel):
    image: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    height: int = 768
    width: int = 1024

    @field_validator("image")
    @classmethod
    def image_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("image must not be empty")
        return value

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt must not be empty")
        return value

    @field_validator("height", "width")
    @classmethod
    def valid_dimension(cls, value: int) -> int:
        return check_dimension(value)


# ============================================================
# Helpers
# ============================================================


def emit_fatal(message: str) -> None:
    print(f"VAST_WORKER_FATAL: {message}", flush=True)
    logger.error(message)


def pick_dtype():
    bf16_ok = device == "cuda" and bool(
        getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    if bf16_ok:
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.float32


def check_dimension(value: int, name: str = "dimension") -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")

    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")

    if value % DIM_MULTIPLE != 0:
        raise ValueError(f"{name} must be a multiple of {DIM_MULTIPLE}")

    if value < MIN_DIM or value > MAX_DIM:
        raise ValueError(f"{name} must be between {MIN_DIM} and {MAX_DIM}")

    return value


def validate_dimension(value: int, name: str = "dimension") -> int:
    try:
        return check_dimension(value, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def validate_prompt(prompt: Optional[str]) -> str:
    if prompt is None or not str(prompt).strip():
        raise HTTPException(
            status_code=400,
            detail="prompt must not be empty",
        )
    return str(prompt).strip()


def decode_base64_image(value: str) -> Image.Image:
    try:
        if "," in value and value.startswith("data:"):
            value = value.split(",", 1)[1]

        raw = base64.b64decode(value, validate=False)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return image

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid base64 image",
        ) from exc


def encode_image_base64(buf: io.BytesIO) -> str:
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def require_model():
    if pipe is None or not model_ready:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready",
        )


# ============================================================
# Model load + warmup
# ============================================================


def load_pipeline_class():
    try:
        from diffusers import Flux2KleinPipeline

        return Flux2KleinPipeline
    except ImportError as exc:
        print(
            f"Flux2KleinPipeline unavailable ({exc}); "
            "trying DiffusionPipeline.from_pretrained",
            flush=True,
        )
        try:
            from diffusers import DiffusionPipeline

            return DiffusionPipeline
        except ImportError as inner:
            emit_fatal(
                "cannot import Flux2KleinPipeline or DiffusionPipeline; "
                f"install diffusers>=0.37.0 ({inner})"
            )
            raise


def load_and_warmup() -> None:
    global pipe, model_ready

    if not torch.cuda.is_available():
        emit_fatal("CUDA is not available; this worker requires a GPU")
        return

    if not os.path.isdir(MODEL_PATH):
        emit_fatal(f"model directory missing: {MODEL_PATH}")
        return

    entries = os.listdir(MODEL_PATH)
    if not entries:
        emit_fatal(f"model directory empty: {MODEL_PATH}")
        return

    dtype = pick_dtype()
    print(
        f"Loading FLUX.2-klein-4B from {MODEL_PATH} "
        f"device={device} dtype={dtype} files={len(entries)}",
        flush=True,
    )
    logger.warning("Loading FLUX.2-klein-4B from %s", MODEL_PATH)

    try:
        pipeline_cls = load_pipeline_class()
        loaded = pipeline_cls.from_pretrained(
            MODEL_PATH,
            torch_dtype=dtype,
            local_files_only=True,
        )
        loaded.to(device)
        pipe = loaded
        print(f"Model moved to {device}", flush=True)
        logger.warning("Model moved to %s", device)
    except Exception as exc:
        emit_fatal(f"model load failed: {exc}")
        traceback.print_exc()
        return

    try:
        print("Warmup run...", flush=True)
        logger.warning("Warmup run...")
        dummy = Image.new("RGB", (512, 512), "white")
        with torch.inference_mode():
            pipe(
                image=dummy,
                prompt="test",
                num_inference_steps=1,
                guidance_scale=GUIDANCE_SCALE,
                height=512,
                width=512,
            )
        torch.cuda.synchronize()
        print("Warmup done", flush=True)
        logger.warning("Warmup done")
    except Exception as exc:
        emit_fatal(f"warmup failed: {exc}")
        traceback.print_exc()
        return

    model_ready = True
    print("VAST_MODEL_READY", flush=True)


def load_and_warmup_safe() -> None:
    try:
        load_and_warmup()
    except Exception as exc:
        emit_fatal(f"model load crashed: {exc}")
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Bind HTTP first. A raise here used to kill uvicorn, so Vast never
    # started PyWorker and sat in model_loading until timeout.
    loop = asyncio.get_running_loop()
    loop.run_in_executor(executor, load_and_warmup_safe)
    yield


# ============================================================
# App
# ============================================================

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors()},
    )


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
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=h,
            width=w,
        ).images[0]

    buf = io.BytesIO()
    result.save(buf, format="PNG", optimize=True)
    buf.seek(0)

    elapsed = time.time() - start
    logger.warning("Inference completed in %.2fs", elapsed)
    return buf


async def run_inference_async(
    image: Image.Image,
    prompt: str,
    height: int,
    width: int,
) -> io.BytesIO:
    require_model()
    loop = asyncio.get_running_loop()

    try:
        return await loop.run_in_executor(
            executor,
            run_inference,
            image,
            prompt,
            height,
            width,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Inference failed",
        ) from exc


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
        "model_ready": model_ready,
        "device": device,
        "model_path": MODEL_PATH,
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
    prompt = validate_prompt(prompt)
    height = validate_dimension(height if height is not None else 768, "height")
    width = validate_dimension(width if width is not None else 1024, "width")

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

    buf = await run_inference_async(image, prompt, height, width)

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
    prompt = validate_prompt(request.prompt)
    height = validate_dimension(request.height, "height")
    width = validate_dimension(request.width, "width")

    buf = await run_inference_async(image, prompt, height, width)
    image_base64 = encode_image_base64(buf)

    return {
        "success": True,
        "image": image_base64,
        "format": "png",
        "width": width,
        "height": height,
    }
