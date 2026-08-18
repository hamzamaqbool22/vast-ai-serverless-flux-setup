#!/bin/bash

set -e

source /venv/main/bin/activate

echo "Starting FLUX FastAPI server..."

rm -f /app/app.log
touch /app/app.log

python3 -m uvicorn main:app \
    --host 127.0.0.1 \
    --port 18000 \
    2>&1 | tee -a /app/app.log &

FASTAPI_PID=$!

echo "FastAPI PID: $FASTAPI_PID"

echo "Waiting for FastAPI..."

for i in {1..120}; do
    if curl -sf http://127.0.0.1:18000/health > /dev/null; then
        echo "FastAPI is ready."
        break
    fi

    sleep 1
done

if ! kill -0 "$FASTAPI_PID" 2>/dev/null; then
    echo "FastAPI process exited. Showing app.log:"
    cat /app/app.log
    exit 1
fi

if ! curl -sf http://127.0.0.1:18000/health > /dev/null; then
    echo "FastAPI did not become ready. Showing app.log:"
    cat /app/app.log
    exit 1
fi

echo "FastAPI is ready."
echo "Starting Vast PyWorker bootstrap..."

exec /app/vast_start_server.sh