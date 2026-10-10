#!/usr/bin/env python3
"""Download the required Flickr8k splits and pinned SigLIP 2 checkpoint."""

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_retrieval.config import load_config
from vision_retrieval.setup_assets import materialize_flickr8k
from vision_retrieval.timing import write_timing_report


DATASET_REPO = "jxie/flickr8k"
DATASET_REVISION = "56f58c967835f7c508d684f36bd7897cca9d7634"
PARQUET_FILES = {
    "validation": "data/validation-00000-of-00001-7025a2b596f14b7b.parquet",
    "test": "data/test-00000-of-00001-42a2661d12c73e48.parquet",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "five_crop.yaml"
    )
    args = parser.parse_args()

    try:
        import pyarrow.parquet as parquet
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError("install the project dependencies before downloading assets") from exc

    config = load_config(args.config)
    download_cache = PROJECT_ROOT / "data" / ".downloads"
    total_started = time.perf_counter()
    steps = {}
    model_cache_dir = config.checkpoint_cache_dir or PROJECT_ROOT / "models" / "huggingface"
    dataset_snapshot = (
        download_cache
        / f"datasets--{DATASET_REPO.replace('/', '--')}"
        / "snapshots"
        / DATASET_REVISION
    )
    dataset_cached_before = all(
        (dataset_snapshot / filename).is_file() for filename in PARQUET_FILES.values()
    )
    model_snapshot = (
        model_cache_dir
        / f"models--{config.checkpoint_id.replace('/', '--')}"
        / "snapshots"
        / config.checkpoint_revision
    )
    model_cached_before = (model_snapshot / "model.safetensors").is_file()
    rows = {}
    for split, filename in PARQUET_FILES.items():
        started = time.perf_counter()
        path = hf_hub_download(
            DATASET_REPO,
            filename,
            revision=DATASET_REVISION,
            repo_type="dataset",
            cache_dir=download_cache,
        )
        steps[f"{split}_download_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        rows[split] = parquet.read_table(path).to_pylist()
        steps[f"{split}_parquet_read_seconds"] = time.perf_counter() - started
        print(f"downloaded {split}: {len(rows[split])} images", flush=True)

    started = time.perf_counter()
    summary = materialize_flickr8k(rows, config.dataset_root)
    steps["dataset_materialization_seconds"] = time.perf_counter() - started
    if summary != {"validation": 1000, "test": 1000}:
        raise ValueError(f"unexpected Flickr8k split counts: {summary}")

    started = time.perf_counter()
    snapshot_path = snapshot_download(
        repo_id=config.checkpoint_id,
        revision=config.checkpoint_revision,
        cache_dir=model_cache_dir,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ],
    )
    steps["model_download_seconds"] = time.perf_counter() - started
    provenance = {
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dataset_files": PARQUET_FILES,
        "splits": summary,
        "checkpoint_id": config.checkpoint_id,
        "checkpoint_revision": config.checkpoint_revision,
        "checkpoint_snapshot": str(Path(snapshot_path).resolve()),
    }
    provenance_path = config.dataset_root / "SOURCE.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    timing = write_timing_report(
        config.output_root / "timing" / "asset_setup.json",
        operation="asset_setup",
        total_seconds=time.perf_counter() - total_started,
        steps=steps,
        details={
            "dataset_cached_before": dataset_cached_before,
            "model_cached_before": model_cached_before,
            "dataset_revision": DATASET_REVISION,
            "checkpoint_revision": config.checkpoint_revision,
            "validation_images": summary["validation"],
            "test_images": summary["test"],
        },
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))
    print(json.dumps(timing, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
