#!/usr/bin/env bash
# Pack the competition data into a tarball for transfer to the A100 box
# (dataset is symlinked locally — tar follows links with -h).
#   bash scripts/pack_data.sh [/tmp/snuai_data.tar.gz]
# then on the A100 either:
#   scp devbox:/tmp/snuai_data.tar.gz . && export SNUAI_DATA_URL=file-not-needed \
#     && mkdir -p data && tar xzf snuai_data.tar.gz -C data
# or serve it over http and: export SNUAI_DATA_URL=http://.../snuai_data.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-/tmp/snuai_data.tar.gz}"
tar chzf "$OUT" -C data train.csv test.csv sample_submission.csv train test
echo "packed -> $OUT ($(du -h "$OUT" | cut -f1))"
