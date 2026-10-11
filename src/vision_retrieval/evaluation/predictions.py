"""Portable prediction records for evaluation and method comparison."""

from dataclasses import asdict, dataclass
import gzip
import json
import math
from pathlib import Path
from typing import Mapping, Sequence, Tuple


@dataclass(frozen=True)
class PredictionRecord:
    query_id: str
    relevant_image_ids: Tuple[str, ...]
    ranked_image_ids: Tuple[str, ...]
    ranked_scores: Tuple[float, ...]
    text_encoding_ms: float
    search_ms: float
    end_to_end_ms: float


@dataclass(frozen=True)
class MethodMetadata:
    method_name: str
    method_version: str
    checkpoint_id: str
    checkpoint_revision: str
    manifest_sha256: str
    parameters: dict
    hardware: dict
    software: dict
    uses_extra_model: bool = False
    uses_training: bool = False
    uses_generated_text: bool = False


def _record_from_dict(values: dict) -> PredictionRecord:
    values = dict(values)
    values["relevant_image_ids"] = tuple(values["relevant_image_ids"])
    values["ranked_image_ids"] = tuple(values["ranked_image_ids"])
    values["ranked_scores"] = tuple(float(score) for score in values["ranked_scores"])
    return PredictionRecord(**values)


def write_predictions(path: Path, records: Sequence[PredictionRecord]) -> None:
    """Atomically write gzip-compressed JSONL predictions."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_predictions(path: Path) -> Tuple[PredictionRecord, ...]:
    """Read gzip-compressed JSONL predictions."""
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(_record_from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid prediction record at line {line_number} in {path}"
                ) from exc
    return tuple(records)


def validate_predictions(
    records: Sequence[PredictionRecord],
    expected_relevance: Mapping[str, Sequence[str]],
    gallery_ids: Sequence[str],
) -> None:
    """Validate that predictions exactly match the canonical benchmark."""
    gallery = tuple(gallery_ids)
    gallery_set = set(gallery)
    if not gallery or len(gallery_set) != len(gallery):
        raise ValueError("canonical gallery must be non-empty and unique")

    seen_queries = set()
    for record in records:
        if record.query_id in seen_queries:
            raise ValueError(f"duplicate query ID in predictions: {record.query_id}")
        seen_queries.add(record.query_id)
    if seen_queries != set(expected_relevance):
        missing = sorted(set(expected_relevance) - seen_queries)
        extra = sorted(seen_queries - set(expected_relevance))
        raise ValueError(f"prediction query set mismatch; missing={missing}, extra={extra}")

    for record in records:
        expected = tuple(expected_relevance[record.query_id])
        if record.relevant_image_ids != expected:
            raise ValueError(f"relevance mismatch for query {record.query_id}")
        if len(record.ranked_image_ids) != len(gallery) or set(record.ranked_image_ids) != gallery_set:
            raise ValueError(
                f"gallery ranking for query {record.query_id} must contain every gallery image exactly once"
            )
        if len(record.ranked_scores) != len(record.ranked_image_ids):
            raise ValueError(f"scores do not align with ranking for query {record.query_id}")
        if any(not math.isfinite(score) for score in record.ranked_scores):
            raise ValueError(f"scores must be finite for query {record.query_id}")
        if any(
            left < right
            for left, right in zip(record.ranked_scores, record.ranked_scores[1:])
        ):
            raise ValueError(f"scores must be sorted descending for query {record.query_id}")
        latencies = (
            record.text_encoding_ms,
            record.search_ms,
            record.end_to_end_ms,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in latencies):
            raise ValueError(
                f"latency values must be finite and non-negative for query {record.query_id}"
            )


def write_method_metadata(path: Path, metadata: MethodMetadata) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)

