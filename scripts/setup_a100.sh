#!/usr/bin/env bash
# One-shot A100 box prep: GPU check + deps + data.
#   bash scripts/setup_a100.sh
# Then: bash scripts/run_a100.sh   (smoke -> full SFT, detaches into tmux)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[setup] GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "[setup] installing deps"
python3 -m pip install -q -r requirements.txt -r requirements-gpu.txt

if [ -f data/train.csv ] && [ -f data/test.csv ]; then
  echo "[setup] data already present at ./data"
else
  echo "[setup] data missing"
  if [ -n "${SNUAI_DATA_URL:-}${SNUAI_KAGGLE_COMP:-}" ]; then
    python3 data_download.py data
  else
    cat <<'EOF'
No data/ and no SNUAI_DATA_URL / SNUAI_KAGGLE_COMP set. Fastest path from the dev box:
  (on dev box)  bash scripts/pack_data.sh                 # -> /tmp/snuai_data.tar.gz
  (from here)   scp devbox:/tmp/snuai_data.tar.gz .
                mkdir -p data && tar xzf snuai_data.tar.gz -C data
Or set SNUAI_DATA_URL=<https url> or SNUAI_KAGGLE_COMP=<slug> and rerun this script.
EOF
    exit 1
  fi
fi

echo "[setup] model: \$SNUAI_MODEL_ID > local prequantized 32B > HF hub fallback (unsloth/Qwen3-VL-32B-Instruct-bnb-4bit)"
echo "[setup] ready. next: bash scripts/run_a100.sh"
