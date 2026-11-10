#!/usr/bin/env bash
set -euo pipefail
if [ ! -x .venv/bin/python ]; then
  echo 'Virtual environment missing. Run: bash setup_mac.sh' >&2
  exit 1
fi
.venv/bin/python -m streamlit run app.py
