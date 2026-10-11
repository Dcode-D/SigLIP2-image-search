"""Built-in methods: the exact cosine baseline and MINER five-crop fusion.

Importing this module registers ``global`` and ``five_crop``. The retrievers themselves
live in :mod:`vision_retrieval.methods.retrievers`; this file only adapts them to the
registry (how to build one from the shared cache, which parameters to report).
"""

import json

from ..config import ExperimentConfig
from ..data.flickr8k import load_split, manifest_sha256
from .registry import BuildContext, MethodSpec, register
from .retrievers import FiveCropRetriever, GlobalRetriever


def load_selected_alpha(config: ExperimentConfig, checkpoint_revision: str) -> float:
    """Read the validation-selected five-crop alpha and verify it matches this run."""
    alpha_path = config.output_root / "parameters" / "five_crop_alpha.json"
    if not alpha_path.is_file():
        raise FileNotFoundError(
            f"selected alpha file not found: {alpha_path}; run scripts/tune_alpha.py first"
        )
    alpha_data = json.loads(alpha_path.read_text(encoding="utf-8"))
    validation_images, validation_queries = load_split(config, "validation")
    if alpha_data.get("validation_manifest_sha256") != manifest_sha256(
        validation_images, validation_queries
    ):
        raise ValueError("selected alpha was not tuned on the configured validation manifest")
    if alpha_data.get("checkpoint_revision") != checkpoint_revision:
        raise ValueError("selected alpha and embedding cache use different checkpoint revisions")
    return float(alpha_data["alpha"])


def _build_global(context: BuildContext):
    cache = context.cache
    return GlobalRetriever(cache.metadata.image_ids, cache.global_embeddings), {}


def _build_five_crop(context: BuildContext):
    cache = context.cache
    alpha = load_selected_alpha(context.config, context.checkpoint_revision)
    retriever = FiveCropRetriever(
        cache.metadata.image_ids,
        cache.global_embeddings,
        cache.crop_embeddings,
        cache.metadata.crop_names,
        alpha,
    )
    return retriever, {"alpha": alpha, "crop_ratio": context.config.crop_ratio}


register(
    MethodSpec(
        name="global",
        label="Global cosine (exact, NumPy)",
        family="exact",
        description=(
            "Baseline shared by every method. One SigLIP 2 vector per image, one per "
            "query; score = cosine similarity computed against all gallery vectors with a "
            "single matrix multiplication, then sorted. Exact by construction."
        ),
        builder=_build_global,
    )
)

register(
    MethodSpec(
        name="five_crop",
        label="MINER five-crop fusion",
        family="five_crop",
        description=(
            "MINER-style multi-crop scoring. Each image also has five fixed 60% crops "
            "(four corners + centre) encoded by the same frozen SigLIP 2. "
            "score = (1 - alpha) * cosine(query, whole image) + alpha * max over crops; "
            "alpha is chosen on the validation split only. Recovers small objects that a "
            "single global embedding under-represents."
        ),
        builder=_build_five_crop,
        uses_crops=True,
    )
)
