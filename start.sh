# #!/bin/bash

# set -euo pipefail

# export PYTHONUNBUFFERED=1

# # Use the Vast image venv. Do not create another Python environment.
# if [ -f /venv/main/bin/activate ]; then
#   source /venv/main/bin/activate
# fi

# echo "Starting FLUX Serverless model server..."

# rm -f /app/app.log
# touch /app/app.log

# cd /app

# # PyWorker is started by Vast's serverless runtime, which injects
# # WORKER_PORT, VAST_TCP_PORT_*, PUBLIC_IPADDR, CONTAINER_ID, and REPORT_ADDR.
# # Starting worker.py from this image without those variables crashes the
# # container in a restart loop (KeyError: WORKER_PORT).
# #
# # This script only starts FastAPI on 127.0.0.1:18000. Logs go to /app/app.log
# # so PyWorker can detect VAST_MODEL_READY.

# exec python3 -m uvicorn main:app \
#     --host 127.0.0.1 \
#     --port 18000 \
#     >> /app/app.log 2>&1


#!/bin/bash

#!/bin/bash

#!/bin/bash

set -e

echo "Starting FLUX model server..."

source /venv/main/bin/activate

exec python3 -m uvicorn main:app \
    --host 127.0.0.1 \
    --port 18000