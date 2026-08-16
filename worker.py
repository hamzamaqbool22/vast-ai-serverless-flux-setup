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


# ============================================================
# Benchmark payload
# ============================================================


def benchmark_generator():
    """
    Small representative request used by Vast to measure
    how quickly this worker can process requests.
    """

    return {
        "image": (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
            "CAQAAAC1HAwCAAAAC0lEQVR42mNk"
            "+g8AAQUBAScY42YAAAAASUVORK5CYII="
        ),
        "prompt": "test",
        "height": 512,
        "width": 512,
    }


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
            # Your GPU runs one image at a time.
            allow_parallel_requests=False,
            # Maximum time a request can wait in the worker queue.
            max_queue_time=120.0,
            # One request = one unit of workload.
            workload_calculator=lambda payload: 1.0,
            benchmark_config=BenchmarkConfig(
                generator=benchmark_generator,
                # Keep this low initially because FLUX inference
                # is relatively expensive.
                runs=3,
                concurrency=1,
            ),
        )
    ],
    log_action_config=LogActionConfig(
        # Our FastAPI process prints this after the model
        # has loaded AND warmup has finished.
        on_load=[
            "VAST_MODEL_READY",
        ],
        # Errors that should mark the worker as failed.
        on_error=[
            "Traceback (most recent call last):",
            "RuntimeError",
            "CUDA out of memory",
        ],
    ),
)


# ============================================================
# Start PyWorker
# ============================================================

if __name__ == "__main__":
    Worker(worker_config).run()
