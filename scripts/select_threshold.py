#!/usr/bin/env python3
"""Select a retrieval score threshold from canonical Flickr8k validation predictions."""

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_retrieval.metrics import select_score_threshold


EXPECTED_VALIDATION_MANIFEST_SHA256 = (
    "d3259e317aba8ae64a3e9ef152253f0fe4031289371bb214ace409a1ea0685b2"
)


def _read_jsonl(path: Path) -> list:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select(
    images_path: Path,
    queries_path: Path,
    predictions_path: Path,
    target_recall: float,
) -> dict:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for threshold selection") from exc

    images = _read_jsonl(images_path)
    queries = _read_jsonl(queries_path)
    predictions = _read_jsonl(predictions_path)
    canonical_images = [
        {"split": row["split"], "image_id": row["image_id"]} for row in images
    ]
    canonical_queries = [
        {
            "split": row["split"],
            "query_id": row["query_id"],
            "relevant_image_id": row["relevant_image_id"],
            "caption_index": row["caption_index"],
            "text": row["text"],
        }
        for row in queries
    ]
    canonical = json.dumps(
        {"images": canonical_images, "queries": canonical_queries},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest_hash = hashlib.sha256(canonical).hexdigest()
    if manifest_hash != EXPECTED_VALIDATION_MANIFEST_SHA256:
        raise ValueError(
            "validation manifest hash mismatch: "
            f"expected {EXPECTED_VALIDATION_MANIFEST_SHA256}, got {manifest_hash}"
        )

    gallery = tuple(row["image_id"] for row in images)
    gallery_set = set(gallery)
    relevance = {row["query_id"]: row["relevant_image_id"] for row in queries}
    by_query = {row["query_id"]: row for row in predictions}
    if len(gallery) != 1000 or len(gallery_set) != 1000:
        raise ValueError("validation gallery must contain 1,000 unique images")
    if len(relevance) != 5000 or set(by_query) != set(relevance):
        raise ValueError("predictions must contain all 5,000 validation queries once")
    if len(by_query) != len(predictions):
        raise ValueError("prediction query IDs must be unique")

    score_matrix = np.empty((len(queries), len(gallery)), dtype=np.float32)
    relevant_indices = np.empty(len(queries), dtype=np.int64)
    for query_index, query in enumerate(queries):
        query_id = query["query_id"]
        row = by_query[query_id]
        if row.get("relevant_image_ids") != [relevance[query_id]]:
            raise ValueError(f"relevance mismatch for {query_id}")
        ranked = row.get("ranked_image_ids")
        scores = row.get("ranked_scores")
        if not isinstance(ranked, list) or len(ranked) != 1000 or set(ranked) != gallery_set:
            raise ValueError(f"{query_id} must rank every gallery image exactly once")
        if not isinstance(scores, list) or len(scores) != 1000:
            raise ValueError(f"{query_id} must provide one score per gallery image")
        numeric_scores = np.asarray(scores, dtype=np.float32)
        if not np.isfinite(numeric_scores).all():
            raise ValueError(f"{query_id} contains a non-finite score")
        if np.any(numeric_scores[:-1] < numeric_scores[1:]):
            raise ValueError(f"scores are not descending for {query_id}")
        if any(
            left_score == right_score and left_id > right_id
            for left_score, right_score, left_id, right_id in zip(
                numeric_scores[:-1], numeric_scores[1:], ranked[:-1], ranked[1:]
            )
        ):
            raise ValueError(f"score ties are not ordered by image ID for {query_id}")
        score_matrix[query_index] = numeric_scores
        relevant_indices[query_index] = ranked.index(relevance[query_id])

    selection = select_score_threshold(score_matrix, relevant_indices, target_recall)
    return {
        **asdict(selection),
        "selection_metric": "target_relevant_recall",
        "tie_breaker": "higher_threshold",
        "validation_manifest_sha256": manifest_hash,
        "query_count": len(queries),
        "gallery_size": len(gallery),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-recall", type=float, default=0.95)
    args = parser.parse_args()
    if not 0.0 < args.target_recall <= 1.0:
        parser.error("--target-recall must be in the interval (0, 1]")
    result = select(
        args.images,
        args.queries,
        args.predictions,
        args.target_recall,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
