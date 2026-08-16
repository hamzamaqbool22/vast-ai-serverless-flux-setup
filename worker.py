import base64
import io

from PIL import Image
from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    BenchmarkConfig,
    LogActionConfig,
)


MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 18000
MODEL_LOG_FILE = "/app/app.log"

DEFAULT_HEIGHT = 768
DEFAULT_WIDTH = 1024
DEFAULT_PIXELS = DEFAULT_HEIGHT * DEFAULT_WIDTH


# ============================================================
# Benchmark payload
# ============================================================


def benchmark_generator():
    """
    Representative request used by Vast to measure how quickly
    this worker can process a typical FLUX edit job.
    """
    image = Image.new("RGB", (DEFAULT_WIDTH, DEFAULT_HEIGHT), (90, 90, 90))
    buf = io.BytesIO()
    image.save(buf, format="PNG")

    return {
        "image": base64.b64encode(buf.getvalue()).decode("utf-8"),
        "prompt": "a photo of a red apple on a wooden table, natural lighting",
        "height": DEFAULT_HEIGHT,
        "width": DEFAULT_WIDTH,
    }


def workload_calculator(payload: dict) -> float:
    height = int(payload.get("height", DEFAULT_HEIGHT))
    width = int(payload.get("width", DEFAULT_WIDTH))
    return (height * width) / DEFAULT_PIXELS


# ============================================================
# Worker configuration
# ============================================================

worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    handlers=[
        HandlerConfig(
            route="/generate",
            allow_parallel_requests=False,
            max_queue_time=120.0,
            workload_calculator=workload_calculator,
            benchmark_config=BenchmarkConfig(
                generator=benchmark_generator,
                runs=3,
                concurrency=1,
            ),
        )
    ],
    log_action_config=LogActionConfig(
        on_load=[
            "VAST_MODEL_READY",
        ],
        on_error=[
            "VAST_WORKER_FATAL",
        ],
    ),
)


# ============================================================
# Start PyWorker
# ============================================================

if __name__ == "__main__":
    Worker(worker_config).run()
