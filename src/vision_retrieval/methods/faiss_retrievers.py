"""Vector-search methods backed by FAISS: exact Flat scan and the IVF inverted-file index.

Both consume the same L2-normalised global SigLIP 2 embeddings as the exact NumPy
baseline. What changes is only *how the nearest vectors are found*:

* ``flat`` - exact inner product against every vector (same answer as NumPy, FAISS kernels).
* ``ivf``  - k-means splits the gallery into ``nlist`` cells; a query is compared only
             with vectors in the ``nprobe`` closest cells. Faster at scale, may miss some.

Because every gallery vector has unit length, inner product equals cosine similarity.
Flat uses inner product directly; IVF is built with the L2 metric (what the index type is
designed for) and its squared distances are mapped back to cosine with cos = 1 - d/2, so
both methods report scores on the same scale.

The shared benchmark contract requires a complete ranking of the whole gallery. IVF can
legitimately fail to fill all k slots when the visited cells hold fewer than k vectors
(FAISS returns id -1); those images are appended with a constant filler score below every
returned score, ordered by image ID, so the output is still a valid, deterministic full
ranking and the "miss" shows up as a lower Recall@K instead of a crash.
"""

from dataclasses import dataclass
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .registry import BuildContext, MethodSpec, register
from .retrievers import SearchResult, _validate_top_k

DEFAULT_SEED = 20261009
FILLER_GAP = 1.0  # filler score = (lowest returned score) - FILLER_GAP


def _faiss():
    # torch and faiss-cpu each bundle their own OpenMP runtime on macOS; loading both in
    # one process aborts with "OMP: Error #15" unless the duplicate is tolerated. The flag
    # must be set before faiss is imported, so it lives here rather than in the scripts.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - exercised only without the dependency
        raise RuntimeError(
            "faiss-cpu is required for FAISS retrieval methods; "
            "install it with `pip install faiss-cpu`"
        ) from exc
    return faiss


def complete_ranking(
    image_ids: Sequence[str],
    found_indices: Sequence[int],
    found_scores: Sequence[float],
) -> Tuple[List[str], List[float]]:
    """Turn a possibly partial FAISS result into a full-gallery ranking.

    ``found_indices`` may contain ``-1`` (unreachable) and, defensively, duplicates.
    Returned ids are unique, scores are non-increasing, and ties are ordered by
    ascending image ID, which is exactly what the shared evaluator validates.
    """
    ids = list(image_ids)
    seen = set()
    ranked_ids: List[str] = []
    ranked_scores: List[float] = []
    for index, score in zip(found_indices, found_scores):
        index = int(index)
        if index < 0 or index >= len(ids) or index in seen:
            continue
        seen.add(index)
        ranked_ids.append(ids[index])
        ranked_scores.append(float(score))

    # FAISS returns results sorted by score already, but enforce the tie rule explicitly.
    order = sorted(range(len(ranked_ids)), key=lambda i: (-ranked_scores[i], ranked_ids[i]))
    ranked_ids = [ranked_ids[i] for i in order]
    ranked_scores = [ranked_scores[i] for i in order]

    missing = sorted(ids[i] for i in range(len(ids)) if i not in seen)
    if missing:
        filler = (min(ranked_scores) if ranked_scores else 0.0) - FILLER_GAP
        ranked_ids.extend(missing)
        ranked_scores.extend([filler] * len(missing))
    return ranked_ids, ranked_scores


@dataclass(frozen=True)
class IndexInfo:
    kind: str
    parameters: Dict[str, Any]
    build_seconds: float
    index_bytes: int
    gallery_size: int
    vector_dimension: int


class FaissRetriever:
    """One FAISS index over the global embeddings; ``kind`` selects the algorithm."""

    KINDS = ("flat", "ivf")

    def __init__(
        self,
        image_ids: Sequence[str],
        embeddings: Any,
        kind: str,
        params: Optional[Dict[str, Any]] = None,
        seed: int = DEFAULT_SEED,
        threads: Optional[int] = 1,
    ) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}; got {kind!r}")
        self.image_ids = tuple(image_ids)
        if len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("image_ids must be unique")
        matrix = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
        if matrix.ndim != 2 or matrix.shape[0] != len(self.image_ids) or matrix.shape[1] <= 0:
            raise ValueError("embeddings must have shape [number_of_images, embedding_dimension]")
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("embeddings must be L2-normalised so inner product equals cosine")
        self.kind = kind
        self.params = dict(params or {})
        self.seed = int(seed)
        self.threads = None if threads is None else int(threads)
        self.vector_dimension = int(matrix.shape[1])
        self._matrix = matrix

        faiss = _faiss()
        self._set_threads(faiss)
        started = time.perf_counter()
        self.index = self._build_index(faiss, matrix)
        build_seconds = time.perf_counter() - started
        self.info = IndexInfo(
            kind=kind,
            parameters=dict(self.params),
            build_seconds=build_seconds,
            index_bytes=int(len(faiss.serialize_index(self.index))),
            gallery_size=len(self.image_ids),
            vector_dimension=self.vector_dimension,
        )

    # ------------------------------------------------------------------ construction

    def _set_threads(self, faiss) -> None:
        if self.threads is not None:
            faiss.omp_set_num_threads(self.threads)

    def _to_scores(self, distances: np.ndarray) -> np.ndarray:
        """FAISS returns inner products for Flat and squared L2 distances for IVF.

        For unit vectors ||q - x||^2 = 2 - 2 cos, so cos = 1 - d / 2; this keeps both
        methods' scores on the same cosine scale as the NumPy baseline.
        """
        if self.kind == "flat":
            return distances
        return 1.0 - 0.5 * distances

    def _build_index(self, faiss, x: np.ndarray):
        builders = {"flat": self._build_flat, "ivf": self._build_ivf}
        index = builders[self.kind](faiss, x)
        index.add(x)
        return index

    def _build_flat(self, faiss, x: np.ndarray):
        return faiss.IndexFlatIP(x.shape[1])

    def _build_ivf(self, faiss, x: np.ndarray):
        nlist = int(self.params.setdefault("nlist", 32))
        nprobe = int(self.params.setdefault("nprobe", 8))
        if nlist <= 0 or nprobe <= 0 or nprobe > nlist:
            raise ValueError("ivf requires 0 < nprobe <= nlist")
        d = x.shape[1]
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist, faiss.METRIC_L2)
        index.cp.seed = self.seed
        index.train(x)
        index.nprobe = nprobe
        return index

    # ------------------------------------------------------------------ search

    def search(self, query_vector: Sequence[float], top_k: Optional[int] = None) -> List[SearchResult]:
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.vector_dimension:
            raise ValueError(f"query vector must have shape ({self.vector_dimension},)")
        gallery_size = len(self.image_ids)
        limit = _validate_top_k(top_k, gallery_size)
        # Full ranking (top_k=None) is what the benchmark contract needs; a small top_k is
        # what a demo needs and is where ANN indexes actually save time.
        k = gallery_size if top_k is None else limit
        faiss = _faiss()
        self._set_threads(faiss)
        distances, indices = self.index.search(query, k)
        scores = self._to_scores(distances)
        ranked_ids, ranked_scores = complete_ranking(self.image_ids, indices[0], scores[0])
        if top_k is not None:
            ranked_ids, ranked_scores = ranked_ids[:limit], ranked_scores[:limit]
        return [
            SearchResult(
                image_id=image_id,
                score=score,
                global_score=score,
                crop_score=None,
                winning_crop=None,
            )
            for image_id, score in zip(ranked_ids, ranked_scores)
        ]

    def describe(self) -> Dict[str, Any]:
        return {
            "index_kind": self.info.kind,
            "index_parameters": self.info.parameters,
            "index_build_seconds": self.info.build_seconds,
            "index_bytes": self.info.index_bytes,
            "gallery_size": self.info.gallery_size,
            "vector_dimension": self.info.vector_dimension,
            "faiss_threads": self.threads,
            "seed": self.seed,
        }


# --------------------------------------------------------------------------- registry



def faiss_builder(kind: str, **params):
    """Create a registry builder for a FAISS index kind with fixed parameters."""

    def build(context: BuildContext):
        cache = context.cache
        retriever = FaissRetriever(cache.metadata.image_ids, cache.global_embeddings, kind, params)
        return retriever, retriever.describe()

    return build


def register_faiss_method(
    name: str,
    label: str,
    description: str,
    kind: str,
    threshold_source: str = "global",
    **params,
) -> MethodSpec:
    """One-liner for adding another FAISS preset (e.g. a different nprobe).

    FAISS methods score on the same cosine scale as the exact ``global`` baseline (Flat
    reproduces it exactly, IVF returns the same cosines for the images it reaches), so by
    default they apply the baseline's validation-selected score threshold instead of
    selecting their own. IVF could not select one anyway: when fewer than the target share
    of relevant images sit in the probed cells, no cutoff reaches the target.
    """
    return register(
        MethodSpec(
            name=name,
            label=label,
            family="faiss",
            description=description,
            builder=faiss_builder(kind, **params),
            parameters={"index_kind": kind, **params},
            threshold_source=threshold_source,
        )
    )


register_faiss_method(
    "faiss_flat",
    "FAISS Flat (exact inner product)",
    "Exact search with FAISS's brute-force IndexFlatIP. Same ranking as the NumPy "
    "baseline; included to show that the library itself does not change the answer, "
    "only the speed of the exact kernels.",
    kind="flat",
)

register_faiss_method(
    "faiss_ivf",
    "FAISS IVF (nlist=32, nprobe=8)",
    "Inverted file index. k-means groups the 1,000 gallery vectors into 32 cells (about "
    "sqrt(N)); at query time the text vector is compared with the 32 centroids and only "
    "the 8 nearest cells are scanned, so about a quarter of the gallery is compared. An "
    "image whose cell is not visited cannot be found: that is the price of the speed-up. "
    "nprobe is the knob: nprobe=32 scans everything and reproduces exact search "
    "(see scripts/sweep_ivf_nprobe.py).",
    kind="ivf",
    nlist=32,
    nprobe=8,
)
