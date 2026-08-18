FROM vastai/pytorch:cuda-12.8.1-auto

WORKDIR /app

# Use the CUDA-matched Python environment shipped with the Vast image.
ENV PATH="/venv/main/bin:${PATH}"
ENV VIRTUAL_ENV="/venv/main"
ENV PYTHONUNBUFFERED=1
ENV SERVERLESS=true
ENV WORKER_PORT=3000
ENV USE_SSL=true

# ============================================================
# Python dependencies for the model/API container
# ============================================================

COPY requirements-api.txt .

RUN . /venv/main/bin/activate && \
    pip install --no-cache-dir -r requirements-api.txt


# ============================================================
# Bake FLUX.2-klein-4B into /models/flux
# (before COPY of app files so start.sh/main.py edits do not
# re-download the model)
# ============================================================

ARG HF_TOKEN

RUN mkdir -p /models/flux && \
    . /venv/main/bin/activate && \
    if [ -n "${HF_TOKEN:-}" ]; then \
        export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"; \
    fi && \
    python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='black-forest-labs/FLUX.2-klein-4B', \
    local_dir='/models/flux' \
)"

# Prevent runtime Hub downloads.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV PYTHONPATH="/app"


# ============================================================
# Application files
# ============================================================

COPY main.py .
COPY worker.py .
COPY start.sh .
COPY vast_start_server.sh /app/vast_start_server.sh

RUN chmod +x /app/start.sh /app/vast_start_server.sh


# ============================================================
# Ports
# FastAPI listens on 127.0.0.1:18000 (do not publish).
# PyWorker listens on WORKER_PORT 3000; Vast maps this publicly.
# ============================================================

EXPOSE 3000


# ============================================================
# Start FastAPI, then Vast PyWorker bootstrap
# SSH launch on the Vast template REPLACES this ENTRYPOINT.
# Prefer Docker launch with empty On-start, or set On-start to /app/start.sh.
# ============================================================

ENTRYPOINT ["/app/start.sh"]
