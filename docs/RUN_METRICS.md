# Producing the comparison metrics (with and without threshold)

End result: `artifacts/reports/summary.md` with the three team methods side by side:
Baseline (`global`), MINER (`five_crop`), FAISS (`faiss_flat`, `faiss_ivf`), each with
complete-ranking metrics, thresholded metrics and a bootstrap difference against the
baseline. Run from the repository root with the project virtualenv active
(`source .venv/bin/activate`). Nothing here needs a GPU; on the M3 Pro each test
evaluation takes 3 to 4 minutes, a validation pass about 1 minute.

Prerequisites (already done if `artifacts/embeddings/{validation,test}` exist):

```bash
python scripts/setup_assets.py && python scripts/prepare_manifest.py
python scripts/build_image_cache.py --split all --batch-size 16
python scripts/tune_alpha.py          # five-crop alpha + five_crop_threshold.json (validation only)
```

## One shot

```bash
bash scripts/run_metrics.sh                # steps 0-6 below, log in artifacts/run_metrics.log
bash scripts/run_metrics.sh --diagnostics  # plus the IVF nprobe sweep and the scaling study
```

The steps below are what the script runs, for doing them one at a time or on Windows.

## Step 0. Sanity check the code

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Step 1. Select the baseline threshold on validation

Writes `artifacts/predictions/validation/global.jsonl.gz` and freezes the cutoff in
`artifacts/parameters/global_threshold.json` (expected ≈ 0.1167: the highest cosine
that keeps 95% of the 5,000 validation relevant images). Latency is irrelevant here, so
one timing run and no warm-up.

```bash
python scripts/evaluate.py --method global --split validation --timing-runs 1 --warmup 0
```

`faiss_flat` and `faiss_ivf` reuse this file (`threshold_source = "global"` in
`src/vision_retrieval/methods/faiss_retrievers.py`): same cosine scale, and IVF cannot select a
cutoff of its own (see step 2). `five_crop` keeps its own file from `tune_alpha.py`.

## Step 2 (optional). Cross-check with the shared selector, and see why IVF borrows

```bash
# must print the same threshold as step 1
python scripts/select_threshold.py \
  --images artifacts/manifests/validation_images.jsonl \
  --queries artifacts/manifests/validation_queries.jsonl \
  --predictions artifacts/predictions/validation/global.jsonl.gz \
  --output artifacts/parameters/global_threshold.crosscheck.json

# prints "relevant image reachable for ~91.7% of queries" and writes no threshold file
python scripts/evaluate.py --method faiss_ivf --split validation --timing-runs 1 --warmup 0
```

## Step 3. Evaluate every method on test, with the threshold applied

Each run writes `artifacts/predictions/<method>.jsonl.gz`, `<method>.metadata.json`,
`artifacts/reports/<method>_metrics.{json,md}` (complete ranking **and** a
"Validation-selected score threshold" section) and `<method>_per_query.csv`.
`--require-threshold` makes a missing threshold file an error instead of a silent skip.

```bash
for m in global five_crop faiss_flat faiss_ivf; do
  python scripts/evaluate.py --method $m --split test --require-threshold
done
```

## Step 4. Validate the prediction files with the shared evaluator

```bash
for m in global five_crop faiss_flat faiss_ivf; do
  python scripts/score_predictions.py \
    --images artifacts/manifests/test_images.jsonl \
    --queries artifacts/manifests/test_queries.jsonl \
    --predictions artifacts/predictions/$m.jsonl.gz | head -3
done
```

## Step 5. Bootstrap differences against the baseline

```bash
for m in five_crop faiss_flat faiss_ivf; do
  python scripts/compare_methods.py \
    --queries artifacts/manifests/test_queries.jsonl \
    --baseline artifacts/predictions/global.jsonl.gz \
    --candidate artifacts/predictions/$m.jsonl.gz \
    --output artifacts/reports/compare_global_vs_$m.json
done
```

## Step 6. Build the comparison summary

```bash
python scripts/summarize_results.py                          # artifacts/reports/summary.md (+ .json)
python scripts/summarize_results.py --output docs/RESULTS.md # refresh the tracked copy for the team
```

`docs/RESULTS.md` is the checked-in snapshot of the last full run; `artifacts/` is ignored by
Git, so it is the only copy collaborators see without rerunning.

## Step 7. Demo with the same thresholds

```bash
python app.py        # http://127.0.0.1:7860
```

The start-up log prints one `[thr]` line per method with the cutoff it loaded (FAISS
methods: `global_threshold.json`). Tick or untick **Apply validation-selected score
threshold** to show a query with and without the cutoff; the status line says how many
of the top 10 passed.

Optional diagnostics for the FAISS story: `python scripts/sweep_ivf_nprobe.py` (the `nprobe`
curve) and `python scripts/benchmark_index_scaling.py --queries 300` (when IVF pays off).

## What to expect

- `faiss_flat` equals `global` in every column, thresholded ones included.
- `faiss_ivf` thresholded recall stays below the 95% target: only ~92% of relevant images
  are in the probed cells, and the threshold cannot bring back what the index never saw.
  Say this when presenting; it is the cost of the index, not a threshold problem.
- `five_crop` has a different cutoff value (fused score scale). Do not compare the cutoff
  numbers; compare precision, average results per query and coverage at the matched 95%
  operating point, as the shared guide says.
