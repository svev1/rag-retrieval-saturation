#!/usr/bin/env bash
# download_data.sh — reproduce data/raw/ from the OpenWebSearch corpus.
#
# Prerequisites:
#   - Docker Desktop running
#   - B2ACCESS / LEXIS AAI account (institution-linked)
#
# Notes:
#   - We use the official owilix Docker container instead of installing
#     owilix natively, because owilix 5.2.0 has a Windows path-separator
#     bug that breaks remote downloads. The container runs Linux internally.
#   - Authentication state is kept in a Docker named volume `owi-state`
#     (not in your Windows home folder) to avoid bind-mount permission
#     issues with token files.
#
# Usage (from project root):
#   bash scripts/download_data.sh

set -euo pipefail

UUID="b510baa6-ebd2-11f0-8c43-02a47ca5d9fd"
IMAGE="opencode.it4i.eu:5050/openwebsearcheu-public/owi-cli:latest"
VOLUME="owi-state"
RAW_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/raw"

mkdir -p "$RAW_DIR"

# 1. Make sure the named volume exists (no-op if already created)
docker volume create "$VOLUME" > /dev/null

# 2. Pull the parquet files (documents + embeddings). First run prompts
#    for B2ACCESS login via browser; subsequent runs reuse cached tokens.
MSYS_NO_PATHCONV=1 docker run --rm \
    -v "${VOLUME}:/home/owi/.owi" \
    "$IMAGE" remote pull "all/id=${UUID}" \
    --files='radboud-val/*.parquet' --yes

# 3. Pull topics + qrels (tiny files)
MSYS_NO_PATHCONV=1 docker run --rm \
    -v "${VOLUME}:/home/owi/.owi" \
    "$IMAGE" remote pull "all/id=${UUID}" \
    --files='radboud-val/*.t*' --yes

# 4. Copy the four files from the Docker volume to data/raw/
MSYS_NO_PATHCONV=1 docker run --rm \
    -v "${VOLUME}:/src" \
    -v "${RAW_DIR}:/dst" \
    alpine cp -v \
    "/src/public/special/${UUID}/radboud-val/topics.tsv" \
    "/src/public/special/${UUID}/radboud-val/qrels.txt" \
    "/src/public/special/${UUID}/radboud-val/subsample.parquet" \
    "/src/public/special/${UUID}/radboud-val/subsample_embeddings.parquet" \
    /dst/

# 5. Verify integrity against MANIFEST.yaml
cd "$RAW_DIR"
sha256sum -c <(awk '
  /sha256:/  { sha=$2 }
  /path:/    { sub(/^data\/raw\//, "", $2); printf "%s  %s\n", sha, $2 }
' ../MANIFEST.yaml)

echo "Done. Files in $RAW_DIR:"
ls -la "$RAW_DIR"
