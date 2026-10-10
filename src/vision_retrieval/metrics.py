"""Method-independent retrieval and latency metrics."""

from dataclasses import dataclass
import math
import statistics
from typing import Optional, Sequence, Set


@dataclass(frozen=True)
class QueryMetrics:
    relevant_rank: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    ndcg_at_10: float


@dataclass(frozen=True)
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p95_ms: float


@dataclass(frozen=True)
class EvaluatedQuery:
    query_id: str
    metrics: QueryMetrics
    text_encoding_ms: float
    search_ms: float
    end_to_end_ms: float


@dataclass(frozen=True)
class MetricReport:
    query_count: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    text_encoding: LatencyStats
    search: LatencyStats
    end_to_end: LatencyStats


def evaluate_ranking(
    ranked_ids: Sequence[str], relevant_ids: Set[str]
) -> QueryMetrics:
    """Evaluate one complete ranking with binary relevance."""
    if not ranked_ids:
        raise ValueError("ranked image IDs must not be empty")
    if not relevant_ids:
        raise ValueError("relevant image IDs must not be empty")
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("ranked image IDs must be unique")

    relevant_ranks = [
        rank
        for rank, image_id in enumerate(ranked_ids, start=1)
        if image_id in relevant_ids
    ]
    if not relevant_ranks:
        raise ValueError("relevant image is missing from the ranking")
    first_rank = relevant_ranks[0]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in relevant_ranks
        if rank <= 10
    )
    ideal_count = min(len(relevant_ids), 10)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return QueryMetrics(
        relevant_rank=first_rank,
        recall_at_1=float(first_rank <= 1),
        recall_at_5=float(first_rank <= 5),
        recall_at_10=float(first_rank <= 10),
        reciprocal_rank=1.0 / first_rank,
        ndcg_at_10=dcg / ideal_dcg,
    )


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower]
        + fraction * (sorted_values[upper] - sorted_values[lower])
    )


def summarize_latencies(values: Sequence[float]) -> LatencyStats:
    """Return mean, p50 and p95 using linear percentile interpolation."""
    if not values:
        raise ValueError("latency values must not be empty")
    numeric = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0.0 for value in numeric):
        raise ValueError("latency values must be finite and non-negative")
    ordered = sorted(numeric)
    return LatencyStats(
        mean_ms=float(statistics.fmean(ordered)),
        p50_ms=_percentile(ordered, 0.50),
        p95_ms=_percentile(ordered, 0.95),
    )


def aggregate_metrics(rows: Sequence[EvaluatedQuery]) -> MetricReport:
    """Aggregate per-query quality and latency measurements."""
    if not rows:
        raise ValueError("evaluated queries must not be empty")
    if len({row.query_id for row in rows}) != len(rows):
        raise ValueError("evaluated query IDs must be unique")
    count = len(rows)
    return MetricReport(
        query_count=count,
        recall_at_1=sum(row.metrics.recall_at_1 for row in rows) / count,
        recall_at_5=sum(row.metrics.recall_at_5 for row in rows) / count,
        recall_at_10=sum(row.metrics.recall_at_10 for row in rows) / count,
        mrr=sum(row.metrics.reciprocal_rank for row in rows) / count,
        ndcg_at_10=sum(row.metrics.ndcg_at_10 for row in rows) / count,
        text_encoding=summarize_latencies(
            [row.text_encoding_ms for row in rows]
        ),
        search=summarize_latencies([row.search_ms for row in rows]),
        end_to_end=summarize_latencies([row.end_to_end_ms for row in rows]),
    )

