#!/usr/bin/env bash
# One command: download the gas data, run every stream, rebuild tables and figures (about 2 minutes on a laptop CPU).
set -euo pipefail
cd "$(dirname "$0")"
python3 -c "import numpy, pandas, scipy, sklearn, matplotlib" 2>/dev/null || {
  echo "Missing packages. Run: pip install -r requirements.txt"; exit 1; }
python3 src/download_data.py
python3 src/run_all.py
python3 src/summarize.py
echo "Done. See results/."
