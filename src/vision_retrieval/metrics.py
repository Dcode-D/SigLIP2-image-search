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


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    target_recall: float
    precision: float
    recall: float
    f1: float
    retained_candidate_count: int
    total_candidate_count: int


@dataclass(frozen=True)
class ThresholdMetricReport:
    threshold: float
    retained_candidate_count: int
    retained_candidate_fraction: float
    queries_with_results: int
    query_coverage: float
    average_results_per_query: float
    relevant_images_retained: int
    candidate_precision: float
    candidate_recall: float
    candidate_f1: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float


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


def select_score_threshold(
    score_matrix, relevant_indices, target_recall: float = 0.95
) -> ThresholdSelection:
    """Choose the highest cutoff retaining the target share of relevant pairs."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for threshold selection") from exc

    scores = np.asarray(score_matrix, dtype=np.float32)
    relevant = np.asarray(relevant_indices, dtype=np.int64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("score_matrix must have shape [queries, images]")
    if relevant.shape != (scores.shape[0],):
        raise ValueError("relevant_indices must contain one index per query")
    if np.any(relevant < 0) or np.any(relevant >= scores.shape[1]):
        raise ValueError("relevant index is outside the score matrix")
    if not np.isfinite(scores).all():
        raise ValueError("score_matrix contains non-finite values")
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in the interval (0, 1]")

    positive_scores = scores[np.arange(scores.shape[0]), relevant]
    target_count = int(math.ceil(target_recall * scores.shape[0]))
    threshold = float(np.sort(positive_scores)[::-1][target_count - 1])
    retained = scores >= threshold
    tp = int(np.sum(positive_scores >= threshold))
    kept = int(np.sum(retained))
    false_positive = kept - tp
    false_negative = scores.shape[0] - tp
    f1_denominator = 2 * tp + false_positive + false_negative
    return ThresholdSelection(
        threshold=threshold,
        target_recall=float(target_recall),
        precision=tp / kept,
        recall=tp / scores.shape[0],
        f1=2 * tp / f1_denominator if f1_denominator else 0.0,
        retained_candidate_count=kept,
        total_candidate_count=int(scores.size),
    )


def evaluate_score_threshold(
    ranked_image_ids: Sequence[Sequence[str]],
    ranked_scores: Sequence[Sequence[float]],
    relevant_image_ids: Sequence[str],
    threshold: float,
) -> ThresholdMetricReport:
    """Evaluate a fixed cutoff while treating filtered relevant images as misses."""
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not ranked_image_ids or len(ranked_image_ids) != len(ranked_scores):
        raise ValueError("ranked IDs and scores must be non-empty and aligned")
    if len(ranked_image_ids) != len(relevant_image_ids):
        raise ValueError("one relevant image ID is required per query")

    retained_total = 0
    covered_queries = 0
    relevant_retained = 0
    retained_ranks = []
    for ids, scores, relevant_id in zip(
        ranked_image_ids, ranked_scores, relevant_image_ids
    ):
        if len(ids) != len(scores):
            raise ValueError("ranked IDs and scores must align for every query")
        kept = sum(float(score) >= threshold for score in scores)
        retained_total += kept
        covered_queries += int(kept > 0)
        try:
            rank = ids.index(relevant_id) + 1
        except (AttributeError, ValueError):
            rank = next(
                (index for index, image_id in enumerate(ids, start=1) if image_id == relevant_id),
                None,
            )
        if rank is not None and rank <= kept:
            relevant_retained += 1
            retained_ranks.append(rank)
        else:
            retained_ranks.append(None)

    query_count = len(retained_ranks)
    false_positive = retained_total - relevant_retained
    false_negative = query_count - relevant_retained
    precision = relevant_retained / retained_total if retained_total else 0.0
    recall = relevant_retained / query_count
    f1_denominator = 2 * relevant_retained + false_positive + false_negative
    f1 = 2 * relevant_retained / f1_denominator if f1_denominator else 0.0
    return ThresholdMetricReport(
        threshold=float(threshold),
        retained_candidate_count=retained_total,
        retained_candidate_fraction=retained_total
        / (query_count * len(ranked_image_ids[0])),
        queries_with_results=covered_queries,
        query_coverage=covered_queries / query_count,
        average_results_per_query=retained_total / query_count,
        relevant_images_retained=relevant_retained,
        candidate_precision=precision,
        candidate_recall=recall,
        candidate_f1=f1,
        recall_at_1=sum(rank is not None and rank <= 1 for rank in retained_ranks)
        / query_count,
        recall_at_5=sum(rank is not None and rank <= 5 for rank in retained_ranks)
        / query_count,
        recall_at_10=sum(rank is not None and rank <= 10 for rank in retained_ranks)
        / query_count,
        mrr=sum(1.0 / rank if rank is not None else 0.0 for rank in retained_ranks)
        / query_count,
        ndcg_at_10=sum(
            1.0 / math.log2(rank + 1) if rank is not None and rank <= 10 else 0.0
            for rank in retained_ranks
        )
        / query_count,
    )

