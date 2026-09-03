#!/usr/bin/env bash
set -euo pipefail
python scripts/reproduce_all.py --with-inference --predictions-root artifacts/local_predictions "$@"
