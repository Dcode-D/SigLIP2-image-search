#!/usr/bin/env python3
"""Run one free-form text query against the cached Flickr8k test gallery.

Any registered method works, e.g.::

    python scripts/query.py "a dog running through grass" --method faiss_ivf --top-k 5
"""

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_retrieval.data.cache import load_embedding_cache, read_cache_metadata
from vision_retrieval.config import load_config
from vision_retrieval.encoder import FrozenSiglip2Encoder
from vision_retrieval.methods import BuildContext, build_method, list_methods


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="free-form text to retrieve images for")
    parser.add_argument("--method", choices=list_methods(), default="five_crop")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment.yaml"
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    config = load_config(args.config)
    cache_path = config.output_root / "embeddings" / "test"
    metadata = read_cache_metadata(cache_path)
    if metadata.checkpoint_id != config.checkpoint_id:
        raise ValueError("embedding cache and configuration use different checkpoints")
    cache = load_embedding_cache(cache_path, metadata)

    encoder = FrozenSiglip2Encoder(
        config.checkpoint_id,
        config.checkpoint_revision,
        device=args.device,
        cache_dir=config.checkpoint_cache_dir,
    )
    built = build_method(args.method, BuildContext(config, cache, encoder.checkpoint_revision))

    started = time.perf_counter()
    query_vector = encoder.encode_texts([args.query])[0]
    encoded = time.perf_counter()
    results = built.retriever.search(query_vector, top_k=args.top_k)
    finished = time.perf_counter()

    print(f"Query:  {args.query}")
    print(f"Method: {built.spec.label}  [{args.method}]  build={built.build_seconds * 1000:.1f} ms")
    print(
        f"Timing: text {1000 * (encoded - started):.1f} ms, "
        f"search {1000 * (finished - encoded):.2f} ms\n"
    )
    for rank, result in enumerate(results, start=1):
        details = f"crop={result.winning_crop}" if result.winning_crop else "global"
        print(
            f"{rank}. score={result.score:.6f}  {details}\n"
            f"   {(config.images_dir / result.image_id).resolve()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
