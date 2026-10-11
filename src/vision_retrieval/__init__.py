"""Frozen SigLIP 2 retrieval research package.

Layout: ``data`` (Flickr8k splits, crops, embedding cache, asset download), ``methods``
(retrievers + the registry every entry point uses), ``evaluation`` (metrics, prediction
format, thresholds); ``config``, ``encoder``, ``demo`` and ``timing`` stay at the root.
The names below are re-exported so ``from vision_retrieval import load_split`` keeps working.
"""

from .config import ExperimentConfig, load_config
from .data import CROP_NAMES, ImageRecord, QueryRecord, crop_boxes, five_crops, load_split, manifest_sha256
from .evaluation import (
    MetricReport,
    PredictionRecord,
    QueryMetrics,
    evaluate_ranking,
    load_threshold,
    select_threshold_from_records,
    threshold_path,
)
from .methods import (
    BuildContext,
    BuiltMethod,
    FiveCropRetriever,
    GlobalRetriever,
    MethodSpec,
    SearchResult,
    build_method,
    get_method,
    list_methods,
    register,
)

__all__ = [
    "BuildContext",
    "BuiltMethod",
    "CROP_NAMES",
    "ExperimentConfig",
    "FiveCropRetriever",
    "GlobalRetriever",
    "ImageRecord",
    "MethodSpec",
    "MetricReport",
    "PredictionRecord",
    "QueryMetrics",
    "QueryRecord",
    "SearchResult",
    "build_method",
    "crop_boxes",
    "evaluate_ranking",
    "five_crops",
    "get_method",
    "list_methods",
    "load_config",
    "load_split",
    "load_threshold",
    "manifest_sha256",
    "register",
    "select_threshold_from_records",
    "threshold_path",
]
