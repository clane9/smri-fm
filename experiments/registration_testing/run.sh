#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/data/openneuro_testing/images"

VERSION="${VERSION:-v1}"
N_THREADS="${N_THREADS:-1}"
OFFSET="${OFFSET:-0}"
LIMIT="${LIMIT:-0}"  # 0 = no limit

OUTPUT_DIR="$SCRIPT_DIR/output/${VERSION}"
LOG_FILE="${OUTPUT_DIR}/results.jsonl"

mkdir -p "$OUTPUT_DIR"

mapfile -t images < <(ls "$INPUT_DIR"/*.nii*)
if [[ "$LIMIT" -gt 0 ]]; then
    images=("${images[@]:$OFFSET:$LIMIT}")
else
    images=("${images[@]:$OFFSET}")
fi

for input in "${images[@]}"; do
    fname="$(basename "$input")"
    output="$OUTPUT_DIR/${fname}"
    if [[ -f $output ]]; then
        continue
    fi
    uv run rigid_registration \
        --version "$VERSION" \
        --n-threads "$N_THREADS" \
        "$input" "$output" \
        | tee -a "$LOG_FILE"
done
