#!/usr/bin/env python3
"""Launch the SigLIP 2 Flickr8k retrieval demo."""

import argparse
from pathlib import Path

from vision_retrieval.demo import create_demo


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "five_crop.yaml"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = create_demo(args.config, device=args.device)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
