"""Embedding cache persistence and compatibility checks."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Tuple


@dataclass(frozen=True)
class CacheMetadata:
    split: str
    checkpoint_id: str
    checkpoint_revision: str
    manifest_sha256: str
    crop_ratio: float
    crop_names: Tuple[str, ...]
    image_ids: Tuple[str, ...]
    vector_dimension: int
    dtype: str = "float32"
    normalized: bool = True


@dataclass(frozen=True)
class EmbeddingCache:
    metadata: CacheMetadata
    global_embeddings: Any
    crop_embeddings: Any


def assert_cache_compatible(expected: CacheMetadata, actual: CacheMetadata) -> None:
    """Raise with the first material cache mismatch."""
    for field in (
        "split",
        "checkpoint_id",
        "checkpoint_revision",
        "manifest_sha256",
        "crop_ratio",
        "crop_names",
        "image_ids",
        "vector_dimension",
        "dtype",
        "normalized",
    ):
        if getattr(expected, field) != getattr(actual, field):
            raise ValueError(
                f"cache metadata mismatch for {field}: "
                f"expected {getattr(expected, field)!r}, got {getattr(actual, field)!r}"
            )


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "NumPy is required for embedding caches; install the project dependencies"
        ) from exc
    return np


def _validate_arrays(cache: EmbeddingCache) -> None:
    np = _require_numpy()
    metadata = cache.metadata
    global_shape = (len(metadata.image_ids), metadata.vector_dimension)
    crop_shape = (len(metadata.image_ids), len(metadata.crop_names), metadata.vector_dimension)
    if cache.global_embeddings.shape != global_shape:
        raise ValueError(
            f"global_embeddings shape must be {global_shape}; got {cache.global_embeddings.shape}"
        )
    if cache.crop_embeddings.shape != crop_shape:
        raise ValueError(
            f"crop_embeddings shape must be {crop_shape}; got {cache.crop_embeddings.shape}"
        )
    for name, array in (
        ("global_embeddings", cache.global_embeddings),
        ("crop_embeddings", cache.crop_embeddings),
    ):
        if str(array.dtype) != metadata.dtype:
            raise ValueError(f"{name} dtype must be {metadata.dtype}; got {array.dtype}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
        if metadata.normalized:
            norms = np.linalg.norm(array, axis=-1)
            if not np.allclose(norms, 1.0, atol=1e-5, rtol=0.0):
                raise ValueError(f"{name} must contain L2-normalized vectors")


def _metadata_dict(metadata: CacheMetadata) -> dict:
    return asdict(metadata)


def _metadata_from_dict(values: dict) -> CacheMetadata:
    values = dict(values)
    values["crop_names"] = tuple(values["crop_names"])
    values["image_ids"] = tuple(values["image_ids"])
    return CacheMetadata(**values)


def read_cache_metadata(path: Path) -> CacheMetadata:
    """Read cache metadata without loading its embedding arrays."""
    root = Path(path)
    try:
        values = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"embedding cache is incomplete: {root}") from exc
    return _metadata_from_dict(values)


def save_embedding_cache(path: Path, cache: EmbeddingCache) -> None:
    """Atomically create a new embedding-cache directory."""
    np = _require_numpy()
    _validate_arrays(cache)
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"embedding cache already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=str(target.parent)))
    try:
        np.save(temporary / "global.npy", cache.global_embeddings, allow_pickle=False)
        np.save(temporary / "crops.npy", cache.crop_embeddings, allow_pickle=False)
        (temporary / "metadata.json").write_text(
            json.dumps(_metadata_dict(cache.metadata), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_embedding_cache(path: Path, expected: CacheMetadata) -> EmbeddingCache:
    """Load a cache only when every compatibility field matches."""
    np = _require_numpy()
    root = Path(path)
    actual = read_cache_metadata(root)
    assert_cache_compatible(expected, actual)
    try:
        global_embeddings = np.load(root / "global.npy", allow_pickle=False)
        crop_embeddings = np.load(root / "crops.npy", allow_pickle=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"embedding cache is incomplete: {root}") from exc
    cache = EmbeddingCache(actual, global_embeddings, crop_embeddings)
    _validate_arrays(cache)
    return cache

