#!/usr/bin/env python3
"""Validate and score predictions from any Flickr8k retrieval method."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


EXPECTED_TEST_MANIFEST_SHA256 = (
    "2d9dda23d52dcfef8cd186bcd3d650b5c2f569317d558e3decbb99894cf25cce"
)


def _read_jsonl(path: Path) -> list:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def _percentile(values: list, percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _latency(values: list) -> dict:
    return {
        "mean_ms": float(statistics.fmean(values)),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
    }


def score(images_path: Path, queries_path: Path, predictions_path: Path) -> dict:
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
    manifest_sha256 = hashlib.sha256(canonical).hexdigest()
    if manifest_sha256 != EXPECTED_TEST_MANIFEST_SHA256:
        raise ValueError(
            "test manifest hash mismatch: "
            f"expected {EXPECTED_TEST_MANIFEST_SHA256}, got {manifest_sha256}"
        )

    gallery = tuple(row["image_id"] for row in images)
    gallery_set = set(gallery)
    if len(gallery) != 1000 or len(gallery_set) != 1000:
        raise ValueError("test gallery must contain exactly 1,000 unique images")

    relevance = {row["query_id"]: row["relevant_image_id"] for row in queries}
    if len(queries) != 5000 or len(relevance) != 5000:
        raise ValueError("test manifest must contain exactly 5,000 unique queries")

    by_query = {}
    for row in predictions:
        query_id = row.get("query_id")
        if query_id in by_query:
            raise ValueError(f"duplicate prediction query ID: {query_id}")
        by_query[query_id] = row
    if set(by_query) != set(relevance):
        missing = sorted(set(relevance) - set(by_query))
        extra = sorted(set(by_query) - set(relevance))
        raise ValueError(f"prediction query mismatch; missing={missing}, extra={extra}")

    ranks = []
    text_times = []
    search_times = []
    end_times = []
    for query_id in relevance:
        row = by_query[query_id]
        relevant_ids = row.get("relevant_image_ids")
        if relevant_ids != [relevance[query_id]]:
            raise ValueError(f"relevance mismatch for {query_id}")

        ranked = row.get("ranked_image_ids")
        scores = row.get("ranked_scores")
        if not isinstance(ranked, list) or len(ranked) != 1000 or set(ranked) != gallery_set:
            raise ValueError(f"{query_id} must rank every gallery image exactly once")
        if not isinstance(scores, list) or len(scores) != 1000:
            raise ValueError(f"{query_id} must provide one score per ranked image")
        numeric_scores = [float(value) for value in scores]
        if any(not math.isfinite(value) for value in numeric_scores):
            raise ValueError(f"{query_id} contains a non-finite score")
        for index in range(999):
            if numeric_scores[index] < numeric_scores[index + 1]:
                raise ValueError(f"scores are not descending for {query_id}")
            if (
                numeric_scores[index] == numeric_scores[index + 1]
                and ranked[index] > ranked[index + 1]
            ):
                raise ValueError(f"score ties are not ordered by image ID for {query_id}")

        latencies = []
        for field in ("text_encoding_ms", "search_ms", "end_to_end_ms"):
            value = float(row[field])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid {field} for {query_id}")
            latencies.append(value)
        text_times.append(latencies[0])
        search_times.append(latencies[1])
        end_times.append(latencies[2])
        ranks.append(ranked.index(relevance[query_id]) + 1)

    count = len(ranks)
    return {
        "query_count": count,
        "recall_at_1": sum(rank <= 1 for rank in ranks) / count,
        "recall_at_5": sum(rank <= 5 for rank in ranks) / count,
        "recall_at_10": sum(rank <= 10 for rank in ranks) / count,
        "mrr": sum(1.0 / rank for rank in ranks) / count,
        "ndcg_at_10": sum(
            1.0 / math.log2(rank + 1) if rank <= 10 else 0.0 for rank in ranks
        )
        / count,
        "text_encoding": _latency(text_times),
        "search": _latency(search_times),
        "end_to_end": _latency(end_times),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = score(args.images, args.queries, args.predictions)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
