#!/usr/bin/env python3
"""Show when approximate indexes pay off: search time and recall vs gallery size.

The shared benchmark gallery has only 1,000 images, where brute force takes a few
milliseconds and an approximate index cannot be faster. This study keeps the *real* SigLIP 2
embeddings and real text queries, but grows the gallery synthetically (mixes of pairs
of real vectors plus a little noise, renormalised) to 10k and 100k vectors, then
measures for FAISS Flat (exact) and IVF (two nprobe settings):

* build time and index size in memory,
* top-10 search latency (mean / p50 / p95, batch size one, everything single-threaded),
* recall@10 against exact search on the *same* gallery (fraction of the exact top-10
  that the approximate index returns).

Results are descriptive (synthetic gallery, one machine) and are written to
``artifacts/reports/index_scaling.{json,md}``. They are not benchmark metrics.

Usage::

    python scripts/benchmark_index_scaling.py                 # sizes 1k, 10k, 100k
    python scripts/benchmark_index_scaling.py --sizes 1000 20000 --queries 200
"""

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Dict, List

# Single-thread everything so NumPy (Accelerate) and FAISS are compared on equal footing
# and so FAISS does not race torch's OpenMP runtime. Must happen before NumPy is imported.
for _var in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_retrieval.data.cache import load_embedding_cache, read_cache_metadata
from vision_retrieval.config import load_config
from vision_retrieval.data.flickr8k import load_split
from vision_retrieval.encoder import FrozenSiglip2Encoder
from vision_retrieval.methods.faiss_retrievers import FaissRetriever

TOP_K = 10


@dataclass(frozen=True)
class Row:
    gallery_size: int
    method: str
    parameters: Dict
    build_seconds: float
    index_mb: float
    search_mean_ms: float
    search_p50_ms: float
    search_p95_ms: float
    recall_at_10_vs_exact: float
    speedup_vs_numpy: float
    speedup_vs_faiss_flat: float


def _percentile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (pos - lo) * (ordered[hi] - ordered[lo])


def grow_gallery(real: np.ndarray, size: int, seed: int, noise: float) -> np.ndarray:
    """Real vectors first, then synthetic ones that stay on the image-embedding manifold.

    Each synthetic vector is a convex mix of two real image embeddings (weight in
    [0.3, 0.7]) plus a little Gaussian noise, renormalised to unit length. Mixing keeps
    the new points in the region where real images live, so a text query's nearest
    neighbours are plausible images rather than noise artefacts (pure additive noise in
    768 dimensions produced outliers that approximate indexes could not reach).
    """
    if size <= real.shape[0]:
        return real[:size].copy()
    rng = np.random.default_rng(seed)
    extra_count = size - real.shape[0]
    left = real[rng.integers(0, real.shape[0], size=extra_count)]
    right = real[rng.integers(0, real.shape[0], size=extra_count)]
    weight = rng.uniform(0.3, 0.7, size=(extra_count, 1)).astype(np.float32)
    extra = weight * left + (1.0 - weight) * right
    extra += noise * rng.standard_normal(extra.shape).astype(np.float32)
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    return np.ascontiguousarray(np.concatenate([real, extra], axis=0), dtype=np.float32)


def method_grid(n: int) -> List[tuple]:
    """Per-size FAISS settings.

    nlist follows the sqrt(N) rule of thumb. Two nprobe operating points are measured:
    the benchmark preset (scan ~25% of the cells, as ``faiss_ivf`` does with 8 of 32) and
    the usual large-scale setting (~10%).
    """
    nlist = max(8, int(math.sqrt(n)))
    return [
        ("faiss_flat", "flat", {}),
        ("faiss_ivf_25pct", "ivf", {"nlist": nlist, "nprobe": max(1, nlist // 4)}),
        ("faiss_ivf_10pct", "ivf", {"nlist": nlist, "nprobe": max(1, nlist // 10)}),
    ]


def exact_top_k(gallery: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    scores = queries @ gallery.T
    return np.argpartition(-scores, k, axis=1)[:, :k]


def time_numpy(gallery: np.ndarray, queries: np.ndarray, k: int) -> List[float]:
    times = []
    for q in queries:
        started = time.perf_counter_ns()
        scores = gallery @ q
        np.argpartition(-scores, k)[:k]
        times.append((time.perf_counter_ns() - started) / 1e6)
    return times


def run(config_path: Path, sizes: List[int], query_count: int, noise: float, seed: int, device: str):
    config = load_config(config_path)
    cache_path = config.output_root / "embeddings" / "test"
    metadata = read_cache_metadata(cache_path)
    cache = load_embedding_cache(cache_path, metadata)
    real = np.ascontiguousarray(cache.global_embeddings, dtype=np.float32)

    _, queries_meta = load_split(config, "test")
    texts = [q.text for q in queries_meta[: query_count]]
    encoder = FrozenSiglip2Encoder(
        config.checkpoint_id, config.checkpoint_revision, device=device, cache_dir=config.checkpoint_cache_dir
    )
    queries = np.concatenate(
        [encoder.encode_texts(texts[i : i + 64]) for i in range(0, len(texts), 64)], axis=0
    ).astype(np.float32)
    print(f"encoded {len(texts)} real test captions as queries", flush=True)

    rows: List[Row] = []
    for size in sizes:
        gallery = grow_gallery(real, size, seed, noise)
        ids = tuple(f"v{i:07d}" for i in range(size))
        exact = exact_top_k(gallery, queries, TOP_K)
        exact_sets = [set(row.tolist()) for row in exact]

        numpy_times = time_numpy(gallery, queries, TOP_K)
        numpy_mean = statistics.fmean(numpy_times)
        rows.append(
            Row(size, "numpy_exact", {}, 0.0, gallery.nbytes / 1e6, numpy_mean,
                _percentile(numpy_times, 0.5), _percentile(numpy_times, 0.95), 1.0, 1.0, float("nan"))
        )
        print(f"[{size:>7}] numpy_exact  search {numpy_mean:7.3f} ms", flush=True)

        flat_mean = None
        for name, kind, params in method_grid(size):
            retriever = FaissRetriever(ids, gallery, kind, params, seed=seed, threads=1)
            times = []
            hits = 0
            for qi, q in enumerate(queries):
                started = time.perf_counter_ns()
                results = retriever.search(q, top_k=TOP_K)
                times.append((time.perf_counter_ns() - started) / 1e6)
                got = {int(r.image_id[1:]) for r in results}
                hits += len(got & exact_sets[qi])
            mean_ms = statistics.fmean(times)
            if kind == "flat":
                flat_mean = mean_ms
            row = Row(
                gallery_size=size,
                method=name,
                parameters=retriever.info.parameters,
                build_seconds=retriever.info.build_seconds,
                index_mb=retriever.info.index_bytes / 1e6,
                search_mean_ms=mean_ms,
                search_p50_ms=_percentile(times, 0.5),
                search_p95_ms=_percentile(times, 0.95),
                recall_at_10_vs_exact=hits / (len(queries) * TOP_K),
                speedup_vs_numpy=numpy_mean / mean_ms if mean_ms > 0 else float("inf"),
                speedup_vs_faiss_flat=(flat_mean / mean_ms) if (flat_mean and mean_ms > 0) else float("nan"),
            )
            rows.append(row)
            print(
                f"[{size:>7}] {name:11s} build {row.build_seconds:6.2f}s  "
                f"{row.index_mb:7.1f} MB  search {mean_ms:7.3f} ms  "
                f"R@10 vs exact {row.recall_at_10_vs_exact:.3f}  x{row.speedup_vs_faiss_flat:.1f} vs flat",
                flush=True,
            )
    return rows, {"queries": len(texts), "noise": noise, "seed": seed, "device": str(encoder.device)}


def write_reports(rows: List[Row], details: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": "Synthetic gallery growth (real vectors + noisy copies). Descriptive only.",
        "hardware": {"platform": platform.platform(), "machine": platform.machine()},
        "details": details,
        "rows": [asdict(r) for r in rows],
    }
    (output_dir / "index_scaling.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Index scaling study (descriptive)",
        "",
        f"Queries: {details['queries']} real Flickr8k test captions · top-{TOP_K} · batch size 1 · "
        f"NumPy and FAISS both pinned to one thread · noise {details['noise']} · {platform.machine()}",
        "",
        "Both brute-force baselines (NumPy matmul, FAISS Flat) and all indexes run single-threaded, "
        "so the last two columns are like-for-like speed-ups of an index over brute force.",
        "",
        "| Gallery | Method | Parameters | Build (s) | Index (MB) | Search mean (ms) | p95 (ms) | R@10 vs exact | vs NumPy | vs FAISS flat |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        params = ", ".join(f"{k}={v}" for k, v in r.parameters.items()) or "-"
        lines.append(
            f"| {r.gallery_size:,} | {r.method} | {params} | {r.build_seconds:.2f} | {r.index_mb:.1f} | "
            f"{r.search_mean_ms:.3f} | {r.search_p95_ms:.3f} | {r.recall_at_10_vs_exact:.3f} | x{r.speedup_vs_numpy:.1f} | "
            + ("-" if math.isnan(r.speedup_vs_faiss_flat) else f"x{r.speedup_vs_faiss_flat:.1f}") + " |"
        )
    (output_dir / "index_scaling.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {output_dir / 'index_scaling.md'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if any(size <= 0 for size in args.sizes) or args.queries <= 0:
        parser.error("--sizes and --queries must be positive")

    rows, details = run(args.config, args.sizes, args.queries, args.noise, args.seed, args.device)
    write_reports(rows, details, load_config(args.config).output_root / "reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
