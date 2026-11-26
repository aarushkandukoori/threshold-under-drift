#!/usr/bin/env bash
# One command: download the gas data, run every stream, rebuild tables and figures.
# The first test takes about 2 minutes on a laptop CPU; the delay and feedback sweep and the
# second follow-up a few more minutes each with 3 to 4 worker processes (set WORKERS to change that).
set -euo pipefail
cd "$(dirname "$0")"
python3 -c "import numpy, pandas, scipy, sklearn, matplotlib" 2>/dev/null || {
  echo "Missing packages. Run: pip install -r requirements.txt"; exit 1; }
python3 src/download_data.py
python3 src/run_all.py
python3 src/summarize.py
python3 src/run_feedback.py
python3 src/summarize_feedback.py
python3 src/run_confirmations.py
python3 src/run_confirmations.py --fresh
python3 src/summarize_confirmations.py
python3 src/check_change_test.py
echo "Done. See results/."
