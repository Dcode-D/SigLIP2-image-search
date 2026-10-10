#!/usr/bin/env python3
"""Create canonical Flickr8k validation and test manifests."""

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_retrieval.config import load_config
from vision_retrieval.data import write_evaluation_manifests
from vision_retrieval.timing import write_timing_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "five_crop.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    started = time.perf_counter()
    metadata = write_evaluation_manifests(config)
    timing = write_timing_report(
        config.output_root / "timing" / "manifest_preparation.json",
        operation="manifest_preparation",
        total_seconds=time.perf_counter() - started,
        steps={},
        details={
            split: {
                "image_count": values["image_count"],
                "query_count": values["query_count"],
                "manifest_sha256": values["manifest_sha256"],
            }
            for split, values in metadata.items()
        },
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(timing, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
