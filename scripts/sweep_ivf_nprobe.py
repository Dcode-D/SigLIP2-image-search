#!/usr/bin/env python3
"""The one knob of the IVF method: nprobe, on the real 1,000-image test gallery.

With ``nlist`` cells, scanning ``nprobe`` of them compares roughly ``nprobe / nlist`` of
the gallery. The script builds the IVF index once per ``nlist`` and sweeps ``nprobe``,
reporting Recall@1 and Recall@10 on the real test captions, overlap of the top-10 with
exact search, and mean search time per query. ``nprobe = nlist`` reproduces exact search.

Diagnostic only: nothing is selected on the test split, the benchmark preset
(nlist=32, nprobe=8) is a fixed library setting.

Usage::

    python scripts/sweep_ivf_nprobe.py                       # nlist 32, nprobe 1..32
    python scripts/sweep_ivf_nprobe.py --nlist 16 --nprobe 1 2 4 8 16 --queries 1000
"""

import argparse
import os
from pathlib import Path
import sys
import time

for _var in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_retrieval.data.cache import load_embedding_cache, read_cache_metadata
from vision_retrieval.config import load_config
from vision_retrieval.data.flickr8k import load_split
from vision_retrieval.encoder import FrozenSiglip2Encoder
from vision_retrieval.methods.faiss_retrievers import FaissRetriever


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--nlist", type=int, default=32)
    parser.add_argument("--nprobe", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--queries", type=int, default=5000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None, help="markdown table (default artifacts/reports/ivf_tradeoff.md)")
    args = parser.parse_args()

    config = load_config(args.config)
    cache_path = config.output_root / "embeddings" / "test"
    cache = load_embedding_cache(cache_path, read_cache_metadata(cache_path))
    gallery = np.ascontiguousarray(cache.global_embeddings, dtype=np.float32)
    ids = cache.metadata.image_ids
    index_of = {image_id: i for i, image_id in enumerate(ids)}

    _, queries = load_split(config, "test")
    queries = queries[: args.queries]
    encoder = FrozenSiglip2Encoder(
        config.checkpoint_id, config.checkpoint_revision, device=args.device, cache_dir=config.checkpoint_cache_dir
    )
    texts = [q.text for q in queries]
    q_matrix = np.concatenate(
        [encoder.encode_texts(texts[i : i + 64]) for i in range(0, len(texts), 64)], axis=0
    )
    relevant = np.asarray([index_of[q.relevant_image_id] for q in queries])
    exact_scores = q_matrix @ gallery.T
    exact_top10 = [set(row.tolist()) for row in np.argsort(-exact_scores, axis=1)[:, :10]]
    print(f"exact R@1 on {len(queries)} queries: {(exact_scores.argmax(1) == relevant).mean():.4f}")

    header = f"{'nlist':>5} {'nprobe':>6} {'scanned':>8} {'R@1':>7} {'R@10':>7} {'top-10 overlap':>14} {'search ms':>9}"
    print(header)
    lines = [
        "# IVF nprobe sweep (real 1,000-image test gallery)",
        "",
        f"{len(queries)} real test captions · nlist={args.nlist} · FAISS single-threaded · exact R@1 "
        f"{(exact_scores.argmax(1) == relevant).mean():.4f}",
        "",
        "| nlist | nprobe | gallery scanned | R@1 | R@10 | top-10 overlap with exact | search mean (ms) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for nprobe in args.nprobe:
        if not 1 <= nprobe <= args.nlist:
            print(f"skip nprobe={nprobe}: must be in [1, nlist]")
            continue
        retriever = FaissRetriever(ids, gallery, "ivf", {"nlist": args.nlist, "nprobe": nprobe}, threads=1)
        hits_at_1 = hits_at_10 = overlap = 0
        started = time.perf_counter()
        for qi, q in enumerate(q_matrix):
            got = [index_of[r.image_id] for r in retriever.search(q, top_k=10)]
            hits_at_1 += int(got[0] == relevant[qi])
            hits_at_10 += int(relevant[qi] in got)
            overlap += len(set(got) & exact_top10[qi])
        mean_ms = (time.perf_counter() - started) / len(queries) * 1000
        n = len(queries)
        print(
            f"{args.nlist:>5} {nprobe:>6} {nprobe / args.nlist:>7.0%} {hits_at_1 / n:>7.4f} "
            f"{hits_at_10 / n:>7.4f} {overlap / (10 * n):>14.3f} {mean_ms:>9.3f}"
        )
        lines.append(
            f"| {args.nlist} | {nprobe} | {nprobe / args.nlist:.0%} | {hits_at_1 / n:.4f} | "
            f"{hits_at_10 / n:.4f} | {overlap / (10 * n):.3f} | {mean_ms:.3f} |"
        )
    output = args.output or (config.output_root / "reports" / "ivf_tradeoff.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
