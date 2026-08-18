#!/bin/bash
#
# Vast Serverless entrypoint.
# FastAPI (127.0.0.1:18000) then PyWorker (WORKER_PORT=3000).
#
# Vast template (pick ONE):
#   1. Preferred: Docker launch so this ENTRYPOINT runs. Leave On-start EMPTY.
#   2. If the template is SSH launch (replaces ENTRYPOINT), On-start MUST be:
#        /app/start.sh
#      Do not paste a custom uvicorn one-liner. A snippet that does not
#      `cd /app` or `--app-dir /app` cannot import main and dies immediately
#      ("FastAPI exited unexpectedly."). PyWorker never starts; Vast then
#      sits in model_loading until timeout.
#
set -e

# Persist env for SSH/tmux sessions if this image is launched in SSH mode.
env >> /etc/environment || true

if [ -f /venv/main/bin/activate ]; then
    source /venv/main/bin/activate
fi

export PYTHONUNBUFFERED=1
export SERVERLESS=true
export WORKER_PORT="${WORKER_PORT:-3000}"
export REPORT_ADDR="${REPORT_ADDR:-https://run.vast.ai}"
export USE_SSL="${USE_SSL:-true}"
export PYTHONPATH="/app${PYTHONPATH:+:$PYTHONPATH}"

cd /app

echo "Starting FLUX FastAPI server from $(pwd)"
echo "python3=$(command -v python3) WORKER_PORT=${WORKER_PORT}"

rm -f /app/app.log
touch /app/app.log

# Stream app.log to stdout so Vast instance logs include the real traceback.
# (uvicorn is redirected to the file; without this, crashes are invisible.)
tail -F /app/app.log &
TAIL_PID=$!

python3 -m uvicorn main:app \
    --app-dir /app \
    --host 127.0.0.1 \
    --port 18000 \
    >> /app/app.log 2>&1 &

FASTAPI_PID=$!

echo "FastAPI PID: $FASTAPI_PID"
echo "Waiting for FastAPI HTTP on 127.0.0.1:18000 (model may still be loading)..."

dump_and_die() {
    echo "===== /app/app.log ====="
    cat /app/app.log || true
    echo "===== end app.log ====="
    kill "$TAIL_PID" 2>/dev/null || true
    exit 1
}

# Wait for the HTTP server to bind. Do not wait for the model.
# PyWorker reports ready later via VAST_MODEL_READY / VAST_WORKER_FATAL.
for _ in $(seq 1 300); do
    if ! kill -0 "$FASTAPI_PID" 2>/dev/null; then
        echo "FastAPI process exited. Showing app.log:"
        dump_and_die
    fi

    if curl -sf http://127.0.0.1:18000/health > /dev/null; then
        echo "FastAPI HTTP is up (model may still be loading)."
        break
    fi

    sleep 1
done

if ! kill -0 "$FASTAPI_PID" 2>/dev/null; then
    echo "FastAPI process exited. Showing app.log:"
    dump_and_die
fi

if ! curl -sf http://127.0.0.1:18000/health > /dev/null; then
    echo "FastAPI did not bind 127.0.0.1:18000. Showing app.log:"
    dump_and_die
fi

echo "Starting Vast PyWorker bootstrap (WORKER_PORT=${WORKER_PORT})..."

# vast_start_server.sh clones PYWORKER_REPO and runs THAT worker.py.
# Vast does not start PyWorker for you on a custom image.
exec /app/vast_start_server.sh
