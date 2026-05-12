#!/usr/bin/env bash
# Phase 0 smoke test (CLAUDE.md §6 Phase 0 step 5 / Exit Gate 0).
#
# Loads Qwen2.5-0.5B-Instruct via MPS, evaluates 50 GSM8K test items,
# asserts pass@1 >= 0.30, writes baseline to results/tables/00_baseline.md.
# Wall-clock target: <20 min on M1 Pro (CLAUDE.md §3 toy budget).
#
# Override via env vars:
#   SMOKE_N=50 SMOKE_MAX_NEW_TOKENS=256 SMOKE_BATCH_SIZE=4
#   SMOKE_PASS_AT_1_THRESHOLD=0.30
#   SMOKE_MODEL=Qwen/Qwen2.5-0.5B-Instruct

set -euo pipefail

# Resolve repo root from this script's location, regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f .venv/bin/activate ]]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
    else
        echo "ERROR: no .venv found. Run 'make install-dev' first." >&2
        exit 2
    fi
fi

MODEL="${SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
N="${SMOKE_N:-50}"
MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-256}"
BATCH_SIZE="${SMOKE_BATCH_SIZE:-4}"
THRESHOLD="${SMOKE_PASS_AT_1_THRESHOLD:-0.30}"
OUTPUT="results/tables/00_baseline.md"

echo "=== Phase 0 smoke ==="
echo "  model     : $MODEL"
echo "  split     : test"
echo "  n         : $N"
echo "  max_new   : $MAX_NEW_TOKENS"
echo "  batch     : $BATCH_SIZE"
echo "  threshold : pass@1 >= $THRESHOLD"
echo "  output    : $OUTPUT"
echo ""

python -m src.eval.harness \
    --model "$MODEL" \
    --split test \
    --n "$N" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --batch_size "$BATCH_SIZE" \
    --seed 0 \
    --output "$OUTPUT" \
    --verbose

# Parse pass@1 from the markdown table line "| pass@1 | 0.XXXX |".
PASS_AT_1="$(awk -F'|' '/^\| pass@1 \|/ {gsub(/[[:space:]]/, "", $3); print $3; exit}' "$OUTPUT")"

if [[ -z "$PASS_AT_1" ]]; then
    echo "ERROR: could not parse pass@1 from $OUTPUT" >&2
    exit 3
fi

echo ""
echo "=== Smoke result ==="
echo "  threshold : pass@1 >= $THRESHOLD"
echo "  measured  : pass@1  = $PASS_AT_1"

# Float comparison via awk (bash doesn't do floats natively).
if awk -v p="$PASS_AT_1" -v t="$THRESHOLD" 'BEGIN { exit !(p+0 >= t+0) }'; then
    echo "  status    : PASS (Exit Gate 0 satisfied)"
    exit 0
else
    echo "  status    : FAIL — pass@1 below threshold"
    exit 1
fi
