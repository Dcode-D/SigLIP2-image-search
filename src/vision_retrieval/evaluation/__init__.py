"""Method-independent evaluation: ranking and latency metrics, the shared prediction
format, and validation-selected score thresholds linked per method."""

from .metrics import (
    EvaluatedQuery,
    MetricReport,
    QueryMetrics,
    ThresholdMetricReport,
    ThresholdSelection,
    aggregate_metrics,
    evaluate_ranking,
    evaluate_score_threshold,
    select_score_threshold,
)
from .predictions import (
    MethodMetadata,
    PredictionRecord,
    read_predictions,
    validate_predictions,
    write_method_metadata,
    write_predictions,
)
from .thresholds import (
    load_threshold,
    reachable_fraction,
    select_threshold_from_records,
    threshold_file_contents,
    threshold_path,
    write_threshold_file,
)

__all__ = [
    "EvaluatedQuery",
    "MethodMetadata",
    "MetricReport",
    "PredictionRecord",
    "QueryMetrics",
    "ThresholdMetricReport",
    "ThresholdSelection",
    "aggregate_metrics",
    "evaluate_ranking",
    "evaluate_score_threshold",
    "load_threshold",
    "reachable_fraction",
    "read_predictions",
    "select_score_threshold",
    "select_threshold_from_records",
    "threshold_file_contents",
    "threshold_path",
    "validate_predictions",
    "write_method_metadata",
    "write_predictions",
    "write_threshold_file",
]
