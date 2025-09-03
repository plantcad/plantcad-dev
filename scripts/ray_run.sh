#!/bin/bash

set -ex
source .venv/bin/activate

# Start Ray; see: https://docs.skypilot.co/en/v0.9.3/examples/training/ray.html
head_ip=$(echo "$SKYPILOT_NODE_IPS" | head -n1)

# Use non-default ports to run alongside SkyPilot's Ray cluster; see:
# https://docs.ray.io/en/latest/ray-core/configure.html
GCS_PORT=6479               # default: 6379
CLIENT_PORT=20001           # default: 10001
DASHBOARD_PORT=8365         # default: 8265
DASHBOARD_AGENT_PORT=52465  # default: 52365
MIN_WORKER_PORT=20002       # default: 10002
MAX_WORKER_PORT=29999       # default: 19999
TEMP_DIR=/tmp/ray_plantcad
PLASMA_DIRECTORY="$HOME/ray_plantcad/plasma"
OBJECT_SPILLING_DIRECTORY="$HOME/ray_plantcad/spill"

# Check if Ray is already running on the expected port
if ps aux | grep ray | grep -E "(--gcs_server_port=$GCS_PORT|--gcs-address=.*:$GCS_PORT)" &> /dev/null; then
  echo "Ray cluster already running on port $GCS_PORT"
# Start it if not
else
  if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
    echo "Starting Ray head node on port $GCS_PORT"
    # Note: --include-dashboard is essential for Thalas/Marin (it's used programatically)
    ray start --head \
      --disable-usage-stats \
      --include-dashboard true \
      --temp-dir $TEMP_DIR \
      --plasma-directory $PLASMA_DIRECTORY \
      --object-spilling-directory $OBJECT_SPILLING_DIRECTORY \
      --port $GCS_PORT \
      --ray-client-server-port $CLIENT_PORT \
      --dashboard-port $DASHBOARD_PORT \
      --dashboard-agent-listen-port $DASHBOARD_AGENT_PORT \
      --min-worker-port $MIN_WORKER_PORT \
      --max-worker-port $MAX_WORKER_PORT
  else
    echo "Starting Ray worker node on port $GCS_PORT"
    ray start \
      --address $head_ip:$GCS_PORT \
      --disable-usage-stats \
      --temp-dir $TEMP_DIR \
      --plasma-directory $PLASMA_DIRECTORY \
      --object-spilling-directory $OBJECT_SPILLING_DIRECTORY \
      --ray-client-server-port $CLIENT_PORT \
      --dashboard-port $DASHBOARD_PORT \
      --dashboard-agent-listen-port $DASHBOARD_AGENT_PORT \
      --min-worker-port $MIN_WORKER_PORT \
      --max-worker-port $MAX_WORKER_PORT
  fi
fi

# Run command on head node only
if [ "$SKYPILOT_NODE_RANK" == "0" ]; then
  export RAY_ADDRESS=$head_ip:$GCS_PORT
  # Execute all arguments passed to this script (if any)
  if [ $# -gt 0 ]; then
    echo "Executing command: $*"
    exec "$@"
  fi
fi
