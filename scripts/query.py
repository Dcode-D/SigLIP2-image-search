#!/usr/bin/env python3
"""Run one free-form text query against the cached Flickr8k test gallery."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_retrieval.cache import load_embedding_cache, read_cache_metadata
from vision_retrieval.config import load_config
from vision_retrieval.encoder import FrozenSiglip2Encoder
from vision_retrieval.retrieval import FiveCropRetriever, GlobalRetriever


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="free-form text to retrieve images for")
    parser.add_argument("--method", choices=("global", "five_crop"), default="five_crop")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "five_crop.yaml"
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    config = load_config(args.config)
    cache_path = config.output_root / "embeddings" / "test"
    metadata = read_cache_metadata(cache_path)
    if metadata.checkpoint_id != config.checkpoint_id:
        raise ValueError("embedding cache and configuration use different checkpoints")
    cache = load_embedding_cache(cache_path, metadata)

    if args.method == "global":
        retriever = GlobalRetriever(metadata.image_ids, cache.global_embeddings)
    else:
        alpha_path = config.output_root / "parameters" / "five_crop_alpha.json"
        alpha = float(json.loads(alpha_path.read_text(encoding="utf-8"))["alpha"])
        retriever = FiveCropRetriever(
            metadata.image_ids,
            cache.global_embeddings,
            cache.crop_embeddings,
            metadata.crop_names,
            alpha,
        )

    encoder = FrozenSiglip2Encoder(
        config.checkpoint_id,
        config.checkpoint_revision,
        cache_dir=config.checkpoint_cache_dir,
    )
    results = retriever.search(encoder.encode_texts([args.query])[0], top_k=args.top_k)
    print(f"Query: {args.query}\nMethod: {args.method}\n")
    for rank, result in enumerate(results, start=1):
        details = f"crop={result.winning_crop}" if result.winning_crop else "global"
        print(
            f"{rank}. score={result.score:.6f}  {details}\n"
            f"   {(config.images_dir / result.image_id).resolve()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
