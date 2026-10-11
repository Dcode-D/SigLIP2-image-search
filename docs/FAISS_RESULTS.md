# FAISS retrieval method — results (run 2026-10-10)

Same frozen `google/siglip2-base-patch16-224` (revision `0ad8c6e0…`), same Flickr8k test
manifest (`2d9dda23…`), same evaluator (`scripts/evaluate.py` / `scripts/score_predictions.py`)
as every other method in this repository. Only the vector-search step differs.

The method is **approximate nearest-neighbour search with FAISS**, shown at two operating
points: `faiss_flat` (exact scan, the control) and `faiss_ivf` (inverted-file index, the
idea). Why only these two, and the answers to likely questions: `docs/METHODS.md`.

Hardware: MacBook Pro M3 Pro, 36 GB, text encoder on MPS, FAISS pinned to one CPU thread,
batch size 1, 3 timing runs per query, 50 warm-up queries. Quality metrics are
machine-independent; latencies are descriptive.

## Benchmark gallery: 1,000 test images, 5,000 caption queries

| Method | R@1 | R@5 | R@10 | MRR | nDCG@10 | Search only, mean (ms) | End-to-end p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `global` (NumPy exact) | 0.7220 | 0.9174 | 0.9606 | 0.8077 | 0.8434 | 0.82 | 10.15 |
| `five_crop` (MINER, alpha 0.5) | **0.7368** | **0.9206** | **0.9658** | **0.8186** | **0.8531** | 1.58 | 10.96 |
| `faiss_flat` (exact IP) | 0.7220 | 0.9174 | 0.9606 | 0.8077 | 0.8434 | 0.92 | 10.15 |
| `faiss_ivf` (nlist=32, nprobe=8) | 0.6802 | 0.8606 | 0.9002 | 0.7589 | 0.7920 | 0.85 | 9.92 |

Reading:

- `faiss_flat` reproduces `global` to the last digit: the library does not change the answer.
- `faiss_ivf` scans about a quarter of the gallery (8 of 32 k-means cells) and pays
  4.2 R@1 points and 6.0 R@10 points for it. On 1,000 images there is nothing to gain: the
  search step is about 1 ms of a ~10 ms query and text encoding dominates, so an approximate
  index can only lose recall here. That is the motivation for the scaling study below.
- IVF index build (k-means, 32 centroids): 0.003 s; index size 3.18 MB vs 3.07 MB for Flat.

Per-method index metadata (build time, bytes, parameters) is in
`artifacts/predictions/<method>.metadata.json`; per-query rows in `artifacts/reports/<method>_per_query.csv`.

## Statistical comparison against `global` (guide Step 6)

Paired, image-clustered bootstrap: 1,000 source images resampled with replacement, all five
captions of a sampled image kept together, 10,000 resamples, seed 20261009, 95% interval =
2.5th/97.5th percentiles of `candidate - baseline`. Produced by `scripts/compare_methods.py`;
JSON in `artifacts/reports/compare_global_vs_<method>.json`.

| Candidate vs `global` | ΔR@1 [95% CI] | ΔR@5 | ΔR@10 | ΔMRR | ΔnDCG@10 | Verdict |
|---|---:|---:|---:|---:|---:|---|
| `faiss_flat` | +0.0000 [+0.0000, +0.0000] | +0.0000 (n.s.) | +0.0000 (n.s.) | +0.0000 (n.s.) | +0.0000 (n.s.) | identical ranking |
| `faiss_ivf` | -0.0418 [-0.0512, -0.0332] | -0.0568 | -0.0604 | -0.0488 | -0.0514 | significantly worse |
| `five_crop` (teammate) | +0.0148 [+0.0062, +0.0236] | +0.0032 (n.s.) | +0.0052 | +0.0109 | +0.0097 | R@1 gain supported |

The IVF interval excludes zero, so its recall loss is a real effect of the index, not
sampling noise.

## Validation-selected score threshold (guide, optional step; reported separately)

Cutoff = highest score that keeps 95% of the relevant images on the *validation* split,
frozen in `artifacts/parameters/<method>_threshold.json`, applied to the test rankings.
`global`, `faiss_flat` and `faiss_ivf` share one cutoff (selected on `global`, same cosine
scale; IVF cannot select its own: only 91.7% of validation relevant images are reachable, so
its own selection collapses to the filler score). `five_crop` has its own (fused-score
scale). Compare the columns at the matched operating point, not the cutoff values.

| Method | Threshold (from) | Relevant retained | Precision | Avg results/query | Queries with ≥1 result | Thr. R@1 | Thr. R@5 | Thr. R@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `global` (NumPy exact) | 0.1167 (`global`) | 0.9516 | 0.1152 | 8.26 | 0.9890 | 0.7170 | 0.9010 | 0.9360 |
| `five_crop` (MINER, alpha 0.5) | 0.1228 (`five_crop`) | 0.9480 | 0.1258 | 7.53 | 0.9900 | 0.7324 | 0.9034 | 0.9372 |
| `faiss_flat` (exact IP) | 0.1167 (`global`) | 0.9516 | 0.1152 | 8.26 | 0.9890 | 0.7170 | 0.9010 | 0.9360 |
| `faiss_ivf` (nlist=32, nprobe=8) | 0.1167 (`global`) | 0.8902 | 0.1173 | 7.59 | 0.9702 | 0.6764 | 0.8468 | 0.8782 |

Reading: at the same 95% validation operating point, MINER returns fewer candidates per
query (7.5 vs 8.3) with slightly higher precision; IVF keeps only 89% of relevant images
on test because ~8% were never in the probed cells.

## The knob: `nprobe` (real 1,000-image gallery, all 5,000 test captions)

`scripts/sweep_ivf_nprobe.py`, `nlist=32`. `nprobe = 32` scans every cell and is exact again.

| nlist | nprobe | gallery scanned | R@1 | R@10 | top-10 overlap with exact | search mean (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 3% | 0.3712 | 0.4804 | 0.416 | 0.129 |
| 32 | 2 | 6% | 0.4914 | 0.6396 | 0.577 | 0.142 |
| 32 | 4 | 12% | 0.6088 | 0.7978 | 0.748 | 0.133 |
| 32 | 8 | 25% | 0.6802 | 0.9002 | 0.895 | 0.141 |
| 32 | 16 | 50% | 0.7190 | 0.9570 | 0.986 | 0.153 |
| 32 | 32 | 100% | 0.7220 | 0.9606 | 1.000 | 0.178 |

## Scaling study: when does IVF pay off? (descriptive, synthetic gallery)

Real SigLIP 2 test embeddings grown to 10k and 100k vectors by mixing pairs of real image
embeddings plus a little noise; 300 real test captions as queries; top-10; NumPy and FAISS
both single-threaded. `nlist = √N`; two operating points: scan ~25% of the cells (the
benchmark preset) and ~10% (the usual large-scale setting). `R@10 vs exact` = share of the
exact top-10 the index returns.

| Gallery | Method | Parameters | Build (s) | Index (MB) | Search mean (ms) | R@10 vs exact | vs FAISS flat |
|---:|---|---|---:|---:|---:|---:|---:|
| 1,000 | faiss_flat | – | 0.00 | 3.1 | 0.113 | 1.000 | 1.0× |
| 1,000 | faiss_ivf_25pct | nlist=31, nprobe=7 | 0.00 | 3.2 | 0.082 | 0.914 | 1.4× |
| 1,000 | faiss_ivf_10pct | nlist=31, nprobe=3 | 0.00 | 3.2 | 0.073 | 0.746 | 1.6× |
| 10,000 | faiss_flat | – | 0.00 | 30.7 | 1.002 | 1.000 | 1.0× |
| 10,000 | faiss_ivf_25pct | nlist=100, nprobe=25 | 0.05 | 31.1 | 0.655 | 0.963 | 1.5× |
| 10,000 | faiss_ivf_10pct | nlist=100, nprobe=10 | 0.05 | 31.1 | 0.550 | 0.886 | 1.8× |
| 100,000 | faiss_flat | – | 0.01 | 307.2 | 10.568 | 1.000 | 1.0× |
| 100,000 | faiss_ivf_25pct | nlist=316, nprobe=79 | 0.87 | 309.0 | 6.748 | 0.991 | 1.6× |
| 100,000 | faiss_ivf_10pct | nlist=316, nprobe=31 | 0.77 | 309.0 | 5.575 | 0.959 | 1.9× |

NumPy brute force (Accelerate) on the same single thread: 0.023 / 0.452 / 3.654 ms at 1k / 10k / 100k, i.e. faster than FAISS Flat on this CPU; the like-for-like comparison is
index vs FAISS Flat.

## Compliance with `docs/SHARED_BENCHMARK_GUIDE.md`

| Guide step | Status for the FAISS method |
|---|---|
| 1. Pinned dataset revision and manifest hashes | Followed: same `jxie/flickr8k` revision; test manifest `2d9dda23…` verified by the evaluator |
| 2. Validation for tuning, test once | Followed: FAISS settings are fixed library rules of thumb (`nlist = √N`, `nprobe = nlist/4`), nothing was selected on test; the full `nprobe` sweep is reported |
| 3. Prediction format (complete 1,000-image ranking, tie order, latency fields) | Followed: all `.jsonl.gz` files pass `scripts/score_predictions.py` as VALID |
| 4. Metrics R@1 / R@5 / R@10 / MRR / nDCG@10 | Followed: evaluator output equals the tables above |
| 5. Latency protocol (batch 1, 50 warm-up, 3 runs, MPS sync) + offline timing files | Followed: `artifacts/timing/*.json`; index build time in each `.metadata.json` |
| 5b. Optional 95%-validation-recall score threshold | Followed: `evaluate.py --split validation` selects `global_threshold.json` (cross-checked with `scripts/select_threshold.py`); FAISS methods apply it on test (table above). Deviation stated: IVF reuses the baseline's cutoff |
| 6. Paired image-clustered bootstrap | Followed: table above |
| 7. Method details (checkpoint revision, parameters, hardware, software versions) | Followed: `artifacts/predictions/<method>.metadata.json` |

## Reproduce

```bash
python scripts/setup_assets.py && python scripts/prepare_manifest.py
python scripts/build_image_cache.py --split all --batch-size 16
python scripts/tune_alpha.py
python scripts/evaluate.py --method global --split validation --timing-runs 1 --warmup 0   # threshold
for m in global five_crop faiss_flat faiss_ivf; do python scripts/evaluate.py --method $m --require-threshold; done
for m in faiss_flat faiss_ivf five_crop; do python scripts/compare_methods.py \
  --queries artifacts/manifests/test_queries.jsonl --baseline artifacts/predictions/global.jsonl.gz \
  --candidate artifacts/predictions/$m.jsonl.gz --output artifacts/reports/compare_global_vs_$m.json; done
python scripts/sweep_ivf_nprobe.py
python scripts/benchmark_index_scaling.py --queries 300
python scripts/summarize_results.py          # artifacts/reports/summary.md, all methods side by side
```

Step-by-step with expected outputs: `docs/RUN_METRICS.md`.

Outputs land in `artifacts/reports/` (gitignored). Method design and how to add one:
`docs/METHODS.md`.

## Caveats to state when presenting

- Scaling galleries above 1,000 are synthetic; the trend (an index helps only when brute force
  is slow) is the point, not the exact numbers.
- Text queries are run without lowercasing (matches the rest of this repository); the SigLIP 2
  paper lowercases, and an earlier experiment gained ~4 R@1 points from `.lower()`. Changing
  this changes every method equally and must be decided for the whole team.
- `nprobe = 8` of 32 is a deliberately visible operating point; the sweep shows the whole curve.

## Appendix: variants measured and dropped

Same benchmark, same day, then unregistered to keep the comparison to two search blocks
(artifacts parked in `artifacts/_dropped/`):

| Variant | R@1 | R@10 | Note |
|---|---:|---:|---|
| IVF (nlist=16, nprobe=2) | 0.5090 | 0.6744 | first preset; too aggressive (scans 1/8) |
| HNSW (M=16, efSearch=32) | 0.7132 | 0.9492 | best approximate result, hard to explain briefly |
| PQ (m=96, 8 bits) | 0.5806 | 0.9120 | 32× memory compression, not needed at this scale |
