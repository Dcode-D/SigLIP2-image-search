"""Exact global and five-crop retrieval."""

from dataclasses import dataclass
import math
from typing import Any, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SearchResult:
    image_id: str
    score: float
    global_score: float
    crop_score: Optional[float]
    winning_crop: Optional[str]


@dataclass(frozen=True)
class AlphaTrial:
    alpha: float
    recall_at_1: float
    mrr: float


@dataclass(frozen=True)
class AlphaSelection:
    alpha: float
    trials: Tuple[AlphaTrial, ...]


def _shape(value: Any) -> tuple:
    if hasattr(value, "shape"):
        return tuple(int(size) for size in value.shape)
    if not isinstance(value, Sequence):
        return ()
    if not value:
        return (0,)
    child_shape = _shape(value[0])
    if any(_shape(child) != child_shape for child in value):
        raise ValueError("embedding arrays must be rectangular")
    return (len(value),) + child_shape


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(float(a) * float(b) for a, b in zip(left, right)))


def _numpy():
    try:
        import numpy as np
    except ImportError:
        return None
    return np


def _validate_top_k(top_k: Optional[int], gallery_size: int) -> int:
    if top_k is None:
        return gallery_size
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    return min(top_k, gallery_size)


class GlobalRetriever:
    def __init__(self, image_ids: Sequence[str], global_embeddings: Any) -> None:
        self.image_ids = tuple(image_ids)
        self.global_embeddings = global_embeddings
        shape = _shape(global_embeddings)
        if len(shape) != 2 or shape[0] != len(self.image_ids) or shape[1] <= 0:
            raise ValueError(
                "global_embeddings must have shape [number_of_images, embedding_dimension]"
            )
        if len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("image_ids must be unique")
        self.vector_dimension = shape[1]

    def _scores(self, query_vector: Sequence[float]):
        if _shape(query_vector) != (self.vector_dimension,):
            raise ValueError(
                f"query vector must have shape ({self.vector_dimension},)"
            )
        np = _numpy()
        if np is not None:
            matrix = np.asarray(self.global_embeddings, dtype=np.float32)
            query = np.asarray(query_vector, dtype=np.float32)
            return [float(value) for value in matrix @ query]
        return [_dot(row, query_vector) for row in self.global_embeddings]

    def search(
        self, query_vector: Sequence[float], top_k: Optional[int] = None
    ) -> list:
        scores = self._scores(query_vector)
        limit = _validate_top_k(top_k, len(self.image_ids))
        order = sorted(
            range(len(self.image_ids)),
            key=lambda index: (-scores[index], self.image_ids[index]),
        )[:limit]
        return [
            SearchResult(
                image_id=self.image_ids[index],
                score=scores[index],
                global_score=scores[index],
                crop_score=None,
                winning_crop=None,
            )
            for index in order
        ]


class FiveCropRetriever(GlobalRetriever):
    def __init__(
        self,
        image_ids: Sequence[str],
        global_embeddings: Any,
        crop_embeddings: Any,
        crop_names: Sequence[str],
        alpha: float,
    ) -> None:
        super().__init__(image_ids, global_embeddings)
        self.crop_embeddings = crop_embeddings
        self.crop_names = tuple(crop_names)
        self.alpha = float(alpha)
        shape = _shape(crop_embeddings)
        expected = (len(self.image_ids), len(self.crop_names), self.vector_dimension)
        if shape != expected:
            raise ValueError(f"crop_embeddings must have shape {expected}; got {shape}")
        if not self.crop_names or len(set(self.crop_names)) != len(self.crop_names):
            raise ValueError("crop_names must be non-empty and unique")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in the interval [0, 1]")

    def _crop_scores(self, query_vector: Sequence[float]):
        if _shape(query_vector) != (self.vector_dimension,):
            raise ValueError(
                f"query vector must have shape ({self.vector_dimension},)"
            )
        np = _numpy()
        if np is not None:
            crops = np.asarray(self.crop_embeddings, dtype=np.float32)
            query = np.asarray(query_vector, dtype=np.float32)
            all_scores = crops @ query
            best_indices = all_scores.argmax(axis=1)
            best_scores = all_scores.max(axis=1)
            return (
                [float(value) for value in best_scores],
                [int(value) for value in best_indices],
            )

        best_scores = []
        best_indices = []
        for image_crops in self.crop_embeddings:
            scores = [_dot(crop, query_vector) for crop in image_crops]
            best_index = max(range(len(scores)), key=lambda index: scores[index])
            best_scores.append(scores[best_index])
            best_indices.append(best_index)
        return best_scores, best_indices

    def search(
        self, query_vector: Sequence[float], top_k: Optional[int] = None
    ) -> list:
        global_scores = self._scores(query_vector)
        crop_scores, crop_indices = self._crop_scores(query_vector)
        scores = [
            (1.0 - self.alpha) * global_score + self.alpha * crop_score
            for global_score, crop_score in zip(global_scores, crop_scores)
        ]
        limit = _validate_top_k(top_k, len(self.image_ids))
        order = sorted(
            range(len(self.image_ids)),
            key=lambda index: (-scores[index], self.image_ids[index]),
        )[:limit]
        return [
            SearchResult(
                image_id=self.image_ids[index],
                score=float(scores[index]),
                global_score=float(global_scores[index]),
                crop_score=float(crop_scores[index]),
                winning_crop=self.crop_names[crop_indices[index]],
            )
            for index in order
        ]


def select_alpha(
    query_vectors: Any,
    relevant_image_ids: Sequence[str],
    image_ids: Sequence[str],
    global_embeddings: Any,
    crop_embeddings: Any,
    crop_names: Sequence[str],
    alpha_grid: Sequence[float],
) -> AlphaSelection:
    """Select alpha by R@1, then MRR, then the smaller alpha."""
    if len(query_vectors) != len(relevant_image_ids) or len(query_vectors) == 0:
        raise ValueError("query vectors and relevance labels must be non-empty and aligned")
    if len(alpha_grid) == 0:
        raise ValueError("alpha_grid must not be empty")
    gallery = set(image_ids)
    unknown = sorted(set(relevant_image_ids) - gallery)
    if unknown:
        raise ValueError(f"relevant images are missing from the gallery: {unknown}")

    np = _numpy()
    if np is not None:
        return _select_alpha_numpy(
            np,
            query_vectors,
            relevant_image_ids,
            image_ids,
            global_embeddings,
            crop_embeddings,
            crop_names,
            alpha_grid,
        )

    trials = []
    for alpha in alpha_grid:
        retriever = FiveCropRetriever(
            image_ids,
            global_embeddings,
            crop_embeddings,
            crop_names,
            alpha,
        )
        first_place = 0
        reciprocal_rank = 0.0
        for query_vector, relevant_image_id in zip(query_vectors, relevant_image_ids):
            results = retriever.search(query_vector)
            rank = next(
                index
                for index, result in enumerate(results, start=1)
                if result.image_id == relevant_image_id
            )
            first_place += int(rank == 1)
            reciprocal_rank += 1.0 / rank
        query_count = len(relevant_image_ids)
        trials.append(
            AlphaTrial(
                alpha=float(alpha),
                recall_at_1=first_place / query_count,
                mrr=reciprocal_rank / query_count,
            )
        )

    best = max(
        trials,
        key=lambda trial: (trial.recall_at_1, trial.mrr, -trial.alpha),
    )
    if not all(math.isfinite(value) for trial in trials for value in (trial.recall_at_1, trial.mrr)):
        raise ValueError("alpha selection produced non-finite metrics")
    return AlphaSelection(alpha=best.alpha, trials=tuple(trials))


def _select_alpha_numpy(
    np,
    query_vectors,
    relevant_image_ids,
    image_ids,
    global_embeddings,
    crop_embeddings,
    crop_names,
    alpha_grid,
) -> AlphaSelection:
    """Vectorized validation scoring without recomputing similarities per alpha."""
    query_matrix = np.asarray(query_vectors, dtype=np.float32)
    global_matrix = np.asarray(global_embeddings, dtype=np.float32)
    crop_tensor = np.asarray(crop_embeddings, dtype=np.float32)
    gallery_size, vector_dimension = global_matrix.shape
    expected_crop_shape = (gallery_size, len(crop_names), vector_dimension)
    if query_matrix.ndim != 2 or query_matrix.shape[1] != vector_dimension:
        raise ValueError(
            f"query vectors must have shape [queries, {vector_dimension}]"
        )
    if crop_tensor.shape != expected_crop_shape:
        raise ValueError(
            f"crop_embeddings must have shape {expected_crop_shape}; got {crop_tensor.shape}"
        )

    global_scores = query_matrix @ global_matrix.T
    crop_scores = np.empty(
        (query_matrix.shape[0], gallery_size), dtype=np.float32
    )
    score_batch_size = 128
    for start in range(0, query_matrix.shape[0], score_batch_size):
        stop = min(start + score_batch_size, query_matrix.shape[0])
        all_crop_scores = np.einsum(
            "qd,ncd->qnc",
            query_matrix[start:stop],
            crop_tensor,
            optimize=True,
        )
        crop_scores[start:stop] = all_crop_scores.max(axis=2)

    image_ids_array = np.asarray(tuple(image_ids), dtype=str)
    id_to_index = {image_id: index for index, image_id in enumerate(image_ids)}
    relevant_indices = np.asarray(
        [id_to_index[image_id] for image_id in relevant_image_ids], dtype=np.int64
    )
    query_indices = np.arange(query_matrix.shape[0])
    lexical_order = np.argsort(image_ids_array, kind="stable")
    lexical_rank = np.empty(gallery_size, dtype=np.int64)
    lexical_rank[lexical_order] = np.arange(gallery_size)

    trials = []
    for alpha in alpha_grid:
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in the interval [0, 1]")
        fused = (1.0 - alpha) * global_scores + alpha * crop_scores
        top_indices = lexical_order[np.argmax(fused[:, lexical_order], axis=1)]
        relevant_scores = fused[query_indices, relevant_indices]
        ranks = 1 + np.sum(fused > relevant_scores[:, None], axis=1)
        tied_before = (fused == relevant_scores[:, None]) & (
            lexical_rank[None, :] < lexical_rank[relevant_indices, None]
        )
        ranks += np.sum(tied_before, axis=1)
        trials.append(
            AlphaTrial(
                alpha=alpha,
                recall_at_1=float(np.mean(top_indices == relevant_indices)),
                mrr=float(np.mean(1.0 / ranks)),
            )
        )

    best = max(
        trials,
        key=lambda trial: (trial.recall_at_1, trial.mrr, -trial.alpha),
    )
    if not all(
        math.isfinite(value)
        for trial in trials
        for value in (trial.recall_at_1, trial.mrr)
    ):
        raise ValueError("alpha selection produced non-finite metrics")
    return AlphaSelection(alpha=best.alpha, trials=tuple(trials))

