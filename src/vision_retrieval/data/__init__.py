"""Everything on disk that the methods consume: Flickr8k splits and manifests,
five-crop geometry, the embedding cache, and the pinned asset download."""

from .cache import (
    CacheMetadata,
    EmbeddingCache,
    load_embedding_cache,
    read_cache_metadata,
    save_embedding_cache,
)
from .crops import CROP_NAMES, crop_boxes, five_crops
from .flickr8k import (
    ImageRecord,
    QueryRecord,
    load_split,
    manifest_sha256,
    validate_no_split_overlap,
    write_evaluation_manifests,
)
from .setup_assets import materialize_flickr8k

__all__ = [
    "CROP_NAMES",
    "CacheMetadata",
    "EmbeddingCache",
    "ImageRecord",
    "QueryRecord",
    "crop_boxes",
    "five_crops",
    "load_embedding_cache",
    "load_split",
    "manifest_sha256",
    "materialize_flickr8k",
    "read_cache_metadata",
    "save_embedding_cache",
    "validate_no_split_overlap",
    "write_evaluation_manifests",
]
