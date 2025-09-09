#!/bin/bash

set -ex
source .venv/bin/activate

# Validate environment variables
if [ -z "$RAY_SETUP_HEAD_NODE_IP" ]; then
  echo "ERROR: RAY_SETUP_HEAD_NODE_IP is not set" >&2
  exit 1
fi

if [ -z "$RAY_SETUP_NODE_RANK" ]; then
  echo "ERROR: RAY_SETUP_NODE_RANK is not set" >&2
  exit 1
fi

if [ -z "$RAY_SETUP_CLUSTER_NAME" ]; then
  echo "ERROR: RAY_SETUP_CLUSTER_NAME is not set" >&2
  exit 1
fi

# Use non-default ports/dirs to avoid conflicts with other Ray clusters; see:
# https://docs.ray.io/en/latest/ray-core/configure.html
GCS_PORT=6479               # default: 6379
CLIENT_PORT=20001           # default: 10001
DASHBOARD_PORT=8365         # default: 8265
DASHBOARD_AGENT_PORT=52465  # default: 52365
MIN_WORKER_PORT=20002       # default: 10002
MAX_WORKER_PORT=29999       # default: 19999
TEMP_DIR=/tmp/ray_${RAY_SETUP_CLUSTER_NAME}
PLASMA_DIRECTORY="$HOME/ray_${RAY_SETUP_CLUSTER_NAME}/plasma"
OBJECT_SPILLING_DIRECTORY="$HOME/ray_${RAY_SETUP_CLUSTER_NAME}/spill"

# Check if Ray is already running on the expected port
if ps aux | grep ray | grep -E "(--gcs_server_port=$GCS_PORT|--gcs-address=.*:$GCS_PORT)" &> /dev/null; then
  echo "Ray cluster already running on port $GCS_PORT"
# Start it if not
else
  mkdir -p $TEMP_DIR
  mkdir -p $PLASMA_DIRECTORY
  mkdir -p $OBJECT_SPILLING_DIRECTORY

  if [ "$RAY_SETUP_NODE_RANK" == "0" ]; then
    echo "Starting Ray head node on port $GCS_PORT"
    # Notes:
    # - `--include-dashboard` is essential for Thalas/Marin (it's used programatically)
    # - The SkyPilot shell session run on cluster launch will send HUP/SIGTERM to child processes
    #   when complete if not isolated in background via nohup, which implies that Ray does not
    #   daemonize itself (or takes a while to do so)
    nohup ray start --head \
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
      --max-worker-port $MAX_WORKER_PORT \
      > $TEMP_DIR/ray_head_start.log 2>&1 &
  else
    echo "Starting Ray worker node on port $GCS_PORT"
    nohup ray start \
      --address $RAY_SETUP_HEAD_NODE_IP:$GCS_PORT \
      --disable-usage-stats \
      --plasma-directory $PLASMA_DIRECTORY \
      --object-spilling-directory $OBJECT_SPILLING_DIRECTORY \
      --ray-client-server-port $CLIENT_PORT \
      --dashboard-port $DASHBOARD_PORT \
      --dashboard-agent-listen-port $DASHBOARD_AGENT_PORT \
      --min-worker-port $MIN_WORKER_PORT \
      --max-worker-port $MAX_WORKER_PORT \
      > $TEMP_DIR/ray_worker_start.log 2>&1 &
  fi
fi

echo "Ray cluster [$RAY_SETUP_CLUSTER_NAME]initialized."
echo "Ray address: $RAY_SETUP_HEAD_NODE_IP:$GCS_PORT"
echo "View dashboard via: ssh -L $DASHBOARD_PORT:localhost:$DASHBOARD_PORT $RAY_SETUP_CLUSTER_NAME -N # (visit http://localhost:$DASHBOARD_PORT)"
