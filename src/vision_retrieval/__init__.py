"""Frozen SigLIP 2 retrieval research package."""

from .config import ExperimentConfig, load_config
from .crops import CROP_NAMES, crop_boxes, five_crops
from .data import ImageRecord, QueryRecord, load_split, manifest_sha256
from .metrics import MetricReport, QueryMetrics, evaluate_ranking
from .predictions import PredictionRecord
from .retrieval import FiveCropRetriever, GlobalRetriever, SearchResult

__all__ = [
    "ExperimentConfig",
    "ImageRecord",
    "MetricReport",
    "PredictionRecord",
    "QueryMetrics",
    "QueryRecord",
    "CROP_NAMES",
    "crop_boxes",
    "five_crops",
    "evaluate_ranking",
    "FiveCropRetriever",
    "GlobalRetriever",
    "SearchResult",
    "load_config",
    "load_split",
    "manifest_sha256",
]

