#!/bin/bash

set -euo pipefail

export PYTHONUNBUFFERED=1

# Use the Vast image venv. Do not create another Python environment.
if [ -f /venv/main/bin/activate ]; then
  source /venv/main/bin/activate
fi

echo "Starting FLUX Serverless worker..."

rm -f /app/app.log
touch /app/app.log

cd /app

echo "Starting FastAPI..."

python3 -m uvicorn main:app \
    --host 127.0.0.1 \
    --port 18000 \
    >> /app/app.log 2>&1 &

API_PID=$!

cleanup() {
  kill "${API_PID}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

echo "FastAPI PID: ${API_PID}"

sleep 2

if ! kill -0 "${API_PID}" 2>/dev/null; then
  echo "VAST_WORKER_FATAL: FastAPI failed to start" >> /app/app.log
  exit 1
fi

echo "Starting Vast PyWorker..."

python3 /app/worker.py
