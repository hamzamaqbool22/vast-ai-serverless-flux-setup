FROM vastai/pytorch:cuda-12.8.1-auto

WORKDIR /app

# Use the CUDA-matched Python environment shipped with the Vast image.
# Do not create a second venv or reinstall PyTorch.
ENV PATH="/venv/main/bin:${PATH}"
ENV VIRTUAL_ENV="/venv/main"
ENV PYTHONUNBUFFERED=1
ENV SERVERLESS=true


# ============================================================
# Python dependencies
# ============================================================

COPY requirements.txt .

RUN . /venv/main/bin/activate && \
    pip install --no-cache-dir -r requirements.txt


# ============================================================
# Application files
# ============================================================

COPY main.py .
COPY worker.py .
COPY start.sh .

RUN chmod +x /app/start.sh


# ============================================================
# Bake FLUX.2-klein-4B into /models/flux
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

# Prevent runtime Hub downloads after the model is already in the image.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1


# ============================================================
# Internal API port
# ============================================================

EXPOSE 18000


# ============================================================
# Start FastAPI + PyWorker
# ============================================================

CMD ["/app/start.sh"]
