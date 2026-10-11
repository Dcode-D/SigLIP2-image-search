#!/usr/bin/env python3
"""Select the five-crop fusion weight on Flickr8k validation queries."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_retrieval.data.cache import (
    CacheMetadata,
    load_embedding_cache,
    read_cache_metadata,
)
from vision_retrieval.config import load_config
from vision_retrieval.data.flickr8k import load_split, manifest_sha256
from vision_retrieval.encoder import FrozenSiglip2Encoder
from vision_retrieval.evaluation.metrics import select_score_threshold
from vision_retrieval.methods.retrievers import select_alpha


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "NumPy is required to tune alpha; install the project dependencies"
        ) from exc
    return np


def _batches(values, size):
    for start in range(0, len(values), size):
        yield start, values[start : start + size]


def tune(config, batch_size: int, device: str = None) -> dict:
    np = _require_numpy()
    images, queries = load_split(config, "validation")
    encoder = FrozenSiglip2Encoder(
        config.checkpoint_id,
        config.checkpoint_revision,
        device=device,
        cache_dir=config.checkpoint_cache_dir,
    )

    cache_path = config.output_root / "embeddings" / "validation"
    actual = read_cache_metadata(cache_path)
    expected = CacheMetadata(
        split="validation",
        checkpoint_id=config.checkpoint_id,
        checkpoint_revision=encoder.checkpoint_revision,
        manifest_sha256=manifest_sha256(images, queries),
        crop_ratio=config.crop_ratio,
        crop_names=tuple(config.crop_names),
        image_ids=tuple(image.image_id for image in images),
        vector_dimension=actual.vector_dimension,
    )
    cache = load_embedding_cache(cache_path, expected)

    query_batches = []
    texts = [query.text for query in queries]
    for start, batch in _batches(texts, batch_size):
        query_batches.append(encoder.encode_texts(batch))
        print(
            f"validation: encoded {start + len(batch)}/{len(texts)} queries",
            flush=True,
        )
    query_vectors = np.concatenate(query_batches, axis=0).astype("float32", copy=False)

    selection = select_alpha(
        query_vectors=query_vectors,
        relevant_image_ids=tuple(query.relevant_image_id for query in queries),
        image_ids=cache.metadata.image_ids,
        global_embeddings=cache.global_embeddings,
        crop_embeddings=cache.crop_embeddings,
        crop_names=cache.metadata.crop_names,
        alpha_grid=config.alpha_grid,
    )
    result = {
        "alpha": selection.alpha,
        "selection_metric": "recall_at_1",
        "tie_breakers": ["mrr", "smaller_alpha"],
        "checkpoint_id": cache.metadata.checkpoint_id,
        "checkpoint_revision": cache.metadata.checkpoint_revision,
        "validation_manifest_sha256": cache.metadata.manifest_sha256,
        "query_count": len(queries),
        "trials": [asdict(trial) for trial in selection.trials],
    }

    output_dir = config.output_root / "parameters"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "five_crop_alpha.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)

    global_scores = query_vectors @ cache.global_embeddings.T
    crop_scores = np.empty_like(global_scores)
    for start, batch in _batches(query_vectors, 128):
        crop_scores[start : start + len(batch)] = np.einsum(
            "qd,ncd->qnc", batch, cache.crop_embeddings, optimize=True
        ).max(axis=2)
    fused_scores = (
        (1.0 - selection.alpha) * global_scores
        + selection.alpha * crop_scores
    )
    image_index = {
        image_id: index for index, image_id in enumerate(cache.metadata.image_ids)
    }
    relevant_indices = np.asarray(
        [image_index[query.relevant_image_id] for query in queries], dtype=np.int64
    )
    threshold_selection = select_score_threshold(
        fused_scores, relevant_indices, config.threshold_target_recall
    )
    threshold_result = {
        **asdict(threshold_selection),
        "selection_metric": "target_relevant_recall",
        "tie_breaker": "higher_threshold",
        "alpha": selection.alpha,
        "checkpoint_id": cache.metadata.checkpoint_id,
        "checkpoint_revision": cache.metadata.checkpoint_revision,
        "validation_manifest_sha256": cache.metadata.manifest_sha256,
        "query_count": len(queries),
        "gallery_size": len(images),
    }
    threshold_path = output_dir / "five_crop_threshold.json"
    threshold_temporary = threshold_path.with_suffix(".json.tmp")
    threshold_temporary.write_text(
        json.dumps(threshold_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    threshold_temporary.replace(threshold_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(threshold_result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiment.yaml",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    tune(load_config(args.config), args.batch_size, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
