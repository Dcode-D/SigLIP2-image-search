#!/usr/bin/env bash
# One-shot metric pipeline: docs/RUN_METRICS.md steps 0-6 (tests, threshold selection,
# test evaluation of every method with and without the threshold, shared-evaluator check,
# bootstrap vs baseline, summary). Stops at the first failure. Log: artifacts/run_metrics.log
#
#   bash scripts/run_metrics.sh                 # all registered methods
#   bash scripts/run_metrics.sh --diagnostics   # also the IVF nprobe sweep and the scaling study
#   PYTHON=.venv/bin/python bash scripts/run_metrics.sh   # without activating the venv
set -o pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
BASELINE="global"
METHODS="global five_crop faiss_flat faiss_ivf"
DIAGNOSTICS=0
[ "${1:-}" = "--diagnostics" ] && DIAGNOSTICS=1
mkdir -p artifacts
LOG="artifacts/run_metrics.log"
: > "$LOG"
FILTER='WARNING clustering|bos_token|eos_token|Loading weights|evaluated [0-9]+/5000'

step() { echo; echo "### $1  $(date '+%H:%M:%S')"; }
run()  { "$@" 2>&1 | grep -vE "$FILTER"; local rc=${PIPESTATUS[0]}; if [ "$rc" -ne 0 ]; then echo "### FAILED (exit $rc): $*"; exit "$rc"; fi; }

{
step "0 unit tests"
run env PYTHONPATH=src "$PY" -m unittest discover -s tests

step "1 select the baseline threshold on validation"
run "$PY" scripts/evaluate.py --method "$BASELINE" --split validation --timing-runs 1 --warmup 0

step "2 cross-check with the shared selector"
run "$PY" scripts/select_threshold.py \
  --images artifacts/manifests/validation_images.jsonl \
  --queries artifacts/manifests/validation_queries.jsonl \
  --predictions artifacts/predictions/validation/${BASELINE}.jsonl.gz \
  --output artifacts/parameters/${BASELINE}_threshold.crosscheck.json

for m in $METHODS; do
  step "3 test evaluation with threshold: $m"
  run "$PY" scripts/evaluate.py --method "$m" --split test --require-threshold
done

for m in $METHODS; do
  step "4 shared evaluator: $m"
  run "$PY" scripts/score_predictions.py \
    --images artifacts/manifests/test_images.jsonl \
    --queries artifacts/manifests/test_queries.jsonl \
    --predictions artifacts/predictions/$m.jsonl.gz
done

for m in $METHODS; do
  [ "$m" = "$BASELINE" ] && continue
  step "5 bootstrap vs $BASELINE: $m"
  run "$PY" scripts/compare_methods.py \
    --queries artifacts/manifests/test_queries.jsonl \
    --baseline artifacts/predictions/${BASELINE}.jsonl.gz \
    --candidate artifacts/predictions/$m.jsonl.gz \
    --output artifacts/reports/compare_${BASELINE}_vs_$m.json
done

step "6 summary"
run "$PY" scripts/summarize_results.py --baseline "$BASELINE"

if [ "$DIAGNOSTICS" -eq 1 ]; then
  step "7a IVF nprobe sweep"
  run "$PY" scripts/sweep_ivf_nprobe.py
  step "7b index scaling study"
  run "$PY" scripts/benchmark_index_scaling.py --queries 300
fi

echo; echo "### ALL DONE $(date '+%H:%M:%S')  -> artifacts/reports/summary.md"
} 2>&1 | tee -a "$LOG"
