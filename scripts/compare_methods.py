#!/usr/bin/env python3
"""Paired, image-clustered bootstrap comparison of prediction files (Step 6 of the guide).

Both files must come from the same canonical test manifest. For every metric the script
reports baseline, candidate, the paired difference ``candidate - baseline`` and its 95%
confidence interval from 10,000 bootstrap resamples that sample *images* with replacement
and keep all five captions of a sampled image together (seed 20261009). An R@1
improvement counts as supported only when the whole interval is above zero.

Usage::

    python scripts/compare_methods.py \
        --queries artifacts/manifests/test_queries.jsonl \
        --baseline artifacts/predictions/global.jsonl.gz \
        --candidate artifacts/predictions/faiss_ivf.jsonl.gz \
        --output artifacts/reports/compare_global_vs_faiss_ivf.json
"""

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

METRICS = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10")
DEFAULT_SEED = 20261009
DEFAULT_SAMPLES = 10_000


def _read_jsonl(path: Path) -> list:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def per_query_metrics(predictions: list, relevance: dict) -> dict:
    """query_id -> dict of the five metrics, computed exactly as score_predictions.py does."""
    out = {}
    for row in predictions:
        query_id = row["query_id"]
        rank = row["ranked_image_ids"].index(relevance[query_id]) + 1
        out[query_id] = {
            "recall_at_1": float(rank <= 1),
            "recall_at_5": float(rank <= 5),
            "recall_at_10": float(rank <= 10),
            "mrr": 1.0 / rank,
            "ndcg_at_10": 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0,
        }
    return out


def paired_bootstrap(
    baseline: dict, candidate: dict, image_of_query: dict, samples: int, seed: int
) -> dict:
    query_ids = sorted(baseline)
    if sorted(candidate) != query_ids:
        raise ValueError("baseline and candidate must contain the same query IDs")
    images = sorted({image_of_query[q] for q in query_ids})
    image_index = {image: i for i, image in enumerate(images)}
    n_images = len(images)
    n_queries = len(query_ids)

    # per-image sums of per-query differences, so one resample is a weighted sum
    diff_by_image = {m: np.zeros(n_images) for m in METRICS}
    queries_per_image = np.zeros(n_images)
    for q in query_ids:
        i = image_index[image_of_query[q]]
        queries_per_image[i] += 1
        for m in METRICS:
            diff_by_image[m][i] += candidate[q][m] - baseline[q][m]

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_images, size=(samples, n_images))
    counts = np.zeros((samples, n_images))
    for s in range(samples):
        counts[s] = np.bincount(draws[s], minlength=n_images)
    total_queries = counts @ queries_per_image  # queries in each resample

    report = {}
    for m in METRICS:
        base_mean = float(np.mean([baseline[q][m] for q in query_ids]))
        cand_mean = float(np.mean([candidate[q][m] for q in query_ids]))
        resampled = (counts @ diff_by_image[m]) / total_queries
        low, high = np.percentile(resampled, [2.5, 97.5])
        report[m] = {
            "baseline": base_mean,
            "candidate": cand_mean,
            "difference": cand_mean - base_mean,
            "ci95_low": float(low),
            "ci95_high": float(high),
            "significant": bool(low > 0.0 or high < 0.0),
        }
    report["_meta"] = {
        "queries": n_queries,
        "images": n_images,
        "bootstrap_samples": samples,
        "seed": seed,
        "resampling_unit": "source image (all captions of a sampled image kept together)",
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True, help="canonical test_queries.jsonl")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    queries = _read_jsonl(args.queries)
    relevance = {q["query_id"]: q["relevant_image_id"] for q in queries}
    image_of_query = {q["query_id"]: q["relevant_image_id"] for q in queries}
    baseline = per_query_metrics(_read_jsonl(args.baseline), relevance)
    candidate = per_query_metrics(_read_jsonl(args.candidate), relevance)
    if set(baseline) != set(relevance) or set(candidate) != set(relevance):
        raise ValueError("prediction files must cover exactly the canonical query set")

    report = paired_bootstrap(baseline, candidate, image_of_query, args.samples, args.seed)
    report["_meta"]["baseline_file"] = str(args.baseline)
    report["_meta"]["candidate_file"] = str(args.candidate)

    print(f"{'metric':12s} {'baseline':>9s} {'candidate':>9s} {'diff':>8s} {'95% CI':>20s}  verdict")
    for m in METRICS:
        r = report[m]
        verdict = "supported" if r["significant"] and r["difference"] > 0 else (
            "worse (significant)" if r["significant"] else "not significant")
        print(
            f"{m:12s} {r['baseline']:9.4f} {r['candidate']:9.4f} {r['difference']:+8.4f} "
            f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]  {verdict}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
