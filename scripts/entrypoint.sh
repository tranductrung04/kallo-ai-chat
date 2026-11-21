#!/usr/bin/env bash
set -euo pipefail
exec uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1
