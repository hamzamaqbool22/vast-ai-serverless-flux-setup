FROM vastai/pytorch:cuda-12.8.1-auto

WORKDIR /app


# ============================================================
# Python dependencies
# ============================================================

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# ============================================================
# Application files
# ============================================================

COPY main.py .
COPY worker.py .
COPY start.sh .


RUN chmod +x /app/start.sh


# ============================================================
# Download FLUX.2-klein-4B INTO THE IMAGE
# ============================================================

RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='black-forest-labs/FLUX.2-klein-4B', \
    local_dir='/workspace/flux' \
)"


# ============================================================
# Internal API port
# ============================================================

EXPOSE 18000


# ============================================================
# Start
# ============================================================

CMD ["/app/start.sh"]