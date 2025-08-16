#!/usr/bin/env bash
set -e
# SageMaker appends "serve" when starting inference containers.
if [ "$1" = "serve" ] || [ -z "$1" ]; then
  exec uvicorn app:app --host 0.0.0.0 --port 8080
else
  exec "$@"
fi
