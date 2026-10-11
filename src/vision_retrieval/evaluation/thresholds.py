"""Validation-selected score thresholds, linked per method.

Shared protocol (``docs/SHARED_BENCHMARK_GUIDE.md``, optional step): on the validation
split keep the highest cutoff that retains at least ``threshold_target_recall`` of the
relevant-image scores, freeze it in ``artifacts/parameters/<method>_threshold.json``,
then apply it to the *test* predictions and report the thresholded metrics next to (never
instead of) the complete-ranking metrics.

Every :class:`~vision_retrieval.methods.MethodSpec` names the method whose threshold it
uses (``spec.threshold_method``):

* ``global`` and ``five_crop`` own theirs (``five_crop`` scores are a fused scale, so they
  need a separate cutoff; ``tune_alpha.py`` writes it, ``evaluate.py --split validation``
  reproduces it);
* the FAISS methods point at ``global``: same cosine scale, and IVF cannot select a
  cutoff of its own because the images in unvisited cells carry a filler score, so when
  fewer than the target share of relevant images are reachable no cutoff hits the target.

The functions here are pure bookkeeping around :func:`~vision_retrieval.metrics.select_score_threshold`.
"""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..config import ExperimentConfig
from .metrics import ThresholdSelection, select_score_threshold
from ..methods.registry import MethodSpec, get_method
from .predictions import PredictionRecord

SELECTION_METRIC = "target_relevant_recall"
TIE_BREAKER = "higher_threshold"


def threshold_path(config: ExperimentConfig, spec: MethodSpec) -> Path:
    """File holding the threshold that applies to ``spec`` (its own or its source's)."""
    owner = get_method(spec.threshold_method)  # KeyError if the source is not registered
    return config.output_root / "parameters" / f"{owner.name}_threshold.json"


def score_matrix_from_records(
    gallery_ids: Sequence[str], records: Sequence[PredictionRecord]
) -> Tuple[np.ndarray, np.ndarray]:
    """Rebuild the ``[queries, gallery]`` score matrix from complete rankings.

    Returns the matrix in gallery order plus the relevant image's column per query, which
    is what :func:`select_score_threshold` consumes.
    """
    index_of = {image_id: index for index, image_id in enumerate(gallery_ids)}
    if len(index_of) != len(gallery_ids):
        raise ValueError("gallery_ids must be unique")
    if not records:
        raise ValueError("records must not be empty")
    matrix = np.empty((len(records), len(gallery_ids)), dtype=np.float32)
    relevant = np.empty(len(records), dtype=np.int64)
    for row, record in enumerate(records):
        if len(record.ranked_image_ids) != len(gallery_ids) or len(record.ranked_scores) != len(gallery_ids):
            raise ValueError(f"{record.query_id}: ranking must cover the whole gallery exactly once")
        if len(record.relevant_image_ids) != 1:
            raise ValueError(f"{record.query_id}: exactly one relevant image is expected")
        try:
            columns = [index_of[image_id] for image_id in record.ranked_image_ids]
            relevant[row] = index_of[record.relevant_image_ids[0]]
        except KeyError as exc:
            raise ValueError(f"{record.query_id}: image {exc} is not in the gallery") from exc
        if len(set(columns)) != len(columns):
            raise ValueError(f"{record.query_id}: ranking repeats an image")
        matrix[row, columns] = np.asarray(record.ranked_scores, dtype=np.float32)
    return matrix, relevant


def select_threshold_from_records(
    records: Sequence[PredictionRecord], gallery_ids: Sequence[str], target_recall: float
) -> ThresholdSelection:
    matrix, relevant = score_matrix_from_records(gallery_ids, records)
    return select_score_threshold(matrix, relevant, target_recall)


def reachable_fraction(records: Sequence[PredictionRecord]) -> float:
    """Share of queries whose relevant image scored strictly above the lowest score of
    its ranking.

    Exact methods give ~1.0. An approximate index pads the images it could not reach
    with one filler score at the bottom, so this is the share of relevant images the
    index can return at all, which bounds any recall target a threshold could meet.
    """
    if not records:
        raise ValueError("records must not be empty")
    reached = 0
    for record in records:
        floor = min(record.ranked_scores)
        rank = record.ranked_image_ids.index(record.relevant_image_ids[0])
        reached += int(record.ranked_scores[rank] > floor)
    return reached / len(records)


def _jsonable(values: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(values, sort_keys=True, default=str))


def threshold_file_contents(
    selection: ThresholdSelection,
    *,
    method_name: str,
    method_parameters: Dict[str, Any],
    checkpoint_id: str,
    checkpoint_revision: str,
    validation_manifest_sha256: str,
    query_count: int,
    gallery_size: int,
) -> Dict[str, Any]:
    """Same shape as the file ``scripts/tune_alpha.py`` writes for ``five_crop``, plus the
    method name and parameters so a stale file is caught at evaluation time."""
    return {
        **asdict(selection),
        "selection_metric": SELECTION_METRIC,
        "tie_breaker": TIE_BREAKER,
        "method_name": method_name,
        "alpha": float(method_parameters.get("alpha", 0.0)),
        "method_parameters": _jsonable(method_parameters),
        "checkpoint_id": checkpoint_id,
        "checkpoint_revision": checkpoint_revision,
        "validation_manifest_sha256": validation_manifest_sha256,
        "query_count": int(query_count),
        "gallery_size": int(gallery_size),
    }


def write_threshold_file(path: Path, contents: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(contents, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_threshold(
    path: Path,
    *,
    validation_manifest_sha256: str,
    checkpoint_revision: str,
    target_recall: float,
    alpha: Optional[float] = None,
) -> float:
    """Read a frozen threshold and refuse one selected under different conditions."""
    if not path.is_file():
        raise FileNotFoundError(
            f"selected threshold not found: {path}; run "
            f"`python scripts/evaluate.py --method {path.stem.replace('_threshold', '')} --split validation` first"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("validation_manifest_sha256") != validation_manifest_sha256:
        raise ValueError(f"{path.name}: threshold does not match the validation manifest")
    if data.get("checkpoint_revision") != checkpoint_revision:
        raise ValueError(f"{path.name}: threshold was selected with a different checkpoint")
    if float(data.get("target_recall", -1.0)) != float(target_recall):
        raise ValueError(f"{path.name}: threshold uses a different target recall")
    if alpha is not None and "alpha" in data and float(data["alpha"]) != float(alpha):
        raise ValueError(f"{path.name}: threshold and retriever use different alpha values")
    threshold = float(data["threshold"])
    if not np.isfinite(threshold):
        raise ValueError(f"{path.name}: threshold is not finite")
    return threshold
