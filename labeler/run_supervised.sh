#!/usr/bin/env bash
# Stopgap process supervisor for server.mjs, for use until
# deploy/labeler-server.service is installed (needs sudo, which this
# session doesn't have). Restarts on crash, same intent as that unit's
# Restart=always -- see the OOM note at the top of server.mjs for why this
# is needed at all.
set -u
cd "$(dirname "$0")"
while true; do
  echo "[run_supervised] starting server.mjs at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  node --env-file=../.env server.mjs
  echo "[run_supervised] server.mjs exited (code $?) at $(date -u +%Y-%m-%dT%H:%M:%SZ) -- restarting in 5s"
  sleep 5
done
