#!/usr/bin/env bash
set -euo pipefail
if command -v python3.12 >/dev/null 2>&1; then
  PY=python3.12
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo 'Install Python 3.12 first (for example from python.org).' >&2
  exit 1
fi
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo 'Dependencies installed. Next run: ollama pull qwen2.5:3b'
echo 'Then run: ./run_mac.sh'
