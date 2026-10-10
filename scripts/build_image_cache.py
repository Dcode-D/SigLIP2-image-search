#!/usr/bin/env python3
"""Build global and five-crop SigLIP 2 embedding caches."""

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_retrieval.cache import CacheMetadata, EmbeddingCache, save_embedding_cache
from vision_retrieval.config import load_config
from vision_retrieval.crops import CROP_NAMES, five_crops
from vision_retrieval.data import load_split, manifest_sha256
from vision_retrieval.encoder import FrozenSiglip2Encoder
from vision_retrieval.timing import write_timing_report


def _require_runtime_dependencies():
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "NumPy and Pillow are required to build image caches; "
            "install the project dependencies"
        ) from exc
    return np, Image


def _load_rgb(Image, path: Path):
    with Image.open(path) as source:
        return source.convert("RGB").copy()


def _batches(values, size):
    for start in range(0, len(values), size):
        yield start, values[start : start + size]


def build_split_cache(config, split: str, batch_size: int, device: str = None) -> Path:
    total_started = time.perf_counter()
    steps = {}
    np, Image = _require_runtime_dependencies()
    if tuple(config.crop_names) != CROP_NAMES:
        raise ValueError(f"configured crop order must be {CROP_NAMES}")
    started = time.perf_counter()
    images, queries = load_split(config, split)
    steps["dataset_loading_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    encoder = FrozenSiglip2Encoder(
        config.checkpoint_id,
        config.checkpoint_revision,
        device=device,
        cache_dir=config.checkpoint_cache_dir,
    )
    steps["model_loading_seconds"] = time.perf_counter() - started

    global_batches = []
    crop_batches = []
    started = time.perf_counter()
    for start, batch in _batches(images, batch_size):
        pil_images = [_load_rgb(Image, record.path) for record in batch]
        global_batch = encoder.encode_images(pil_images)
        named_crops = [five_crops(image, config.crop_ratio) for image in pil_images]
        flat_crops = [crop for image_crops in named_crops for _, crop in image_crops]
        crop_batch = encoder.encode_images(flat_crops).reshape(
            len(batch), len(CROP_NAMES), -1
        )
        global_batches.append(global_batch)
        crop_batches.append(crop_batch)
        print(f"{split}: encoded {start + len(batch)}/{len(images)} images", flush=True)
    steps["image_loading_cropping_and_encoding_seconds"] = (
        time.perf_counter() - started
    )

    started = time.perf_counter()
    global_embeddings = np.concatenate(global_batches, axis=0).astype("float32", copy=False)
    crop_embeddings = np.concatenate(crop_batches, axis=0).astype("float32", copy=False)
    metadata = CacheMetadata(
        split=split,
        checkpoint_id=config.checkpoint_id,
        checkpoint_revision=encoder.checkpoint_revision,
        manifest_sha256=manifest_sha256(images, queries),
        crop_ratio=config.crop_ratio,
        crop_names=CROP_NAMES,
        image_ids=tuple(record.image_id for record in images),
        vector_dimension=int(global_embeddings.shape[1]),
    )
    output_path = config.output_root / "embeddings" / split
    save_embedding_cache(
        output_path,
        EmbeddingCache(metadata, global_embeddings, crop_embeddings),
    )
    steps["array_assembly_and_cache_write_seconds"] = time.perf_counter() - started
    timing = write_timing_report(
        config.output_root / "timing" / f"image_cache_{split}.json",
        operation="image_cache_construction",
        total_seconds=time.perf_counter() - total_started,
        steps=steps,
        details={
            "split": split,
            "image_count": len(images),
            "batch_size": batch_size,
            "device": str(encoder.device),
            "checkpoint_id": config.checkpoint_id,
            "checkpoint_revision": encoder.checkpoint_revision,
            "crop_ratio": config.crop_ratio,
            "embeddings_per_image": 1 + len(CROP_NAMES),
        },
    )
    print(
        f"{split}: saved global {global_embeddings.shape} and crops "
        f"{crop_embeddings.shape} to {output_path}",
        flush=True,
    )
    print(f"{split}: timing saved to {config.output_root / 'timing' / f'image_cache_{split}.json'}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "five_crop.yaml",
    )
    parser.add_argument(
        "--split", choices=("validation", "test", "all"), default="all"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    config = load_config(args.config)
    splits = ("validation", "test") if args.split == "all" else (args.split,)
    for split in splits:
        build_split_cache(config, split, args.batch_size, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
