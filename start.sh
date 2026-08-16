#!/bin/bash

set -e

echo "Starting FLUX Serverless worker..."

# Make sure we don't accidentally use an old log file.
rm -f /app/app.log
touch /app/app.log

echo "Starting FastAPI..."

python3 -m uvicorn main:app \
    --host 127.0.0.1 \
    --port 18000 \
    >> /app/app.log 2>&1 &

API_PID=$!

echo "FastAPI PID: $API_PID"

# Give Python a moment to start.
sleep 2

echo "Starting Vast PyWorker..."

exec python3 /app/worker.py