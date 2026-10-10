"""Experiment configuration loading and validation."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ExperimentConfig:
    checkpoint_id: str
    checkpoint_revision: Optional[str]
    checkpoint_cache_dir: Optional[Path]
    dataset_root: Path
    images_dir: Path
    captions_file: Path
    split_files: Dict[str, Path]
    output_root: Path
    expected_images_per_split: int
    captions_per_image: int
    crop_ratio: float
    crop_names: Tuple[str, ...]
    alpha_grid: Tuple[float, ...]
    seed: int
    cutoffs: Tuple[int, ...]


def _read_mapping(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is not JSON-compatible YAML; install PyYAML to parse it"
            ) from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"configuration root must be a mapping: {path}")
        return loaded


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(path: Path) -> ExperimentConfig:
    """Load a JSON-compatible YAML configuration relative to its directory."""
    config_path = Path(path).resolve()
    raw = _read_mapping(config_path)

    try:
        checkpoint = raw["checkpoint"]
        dataset = raw["dataset"]
        method = raw["method"]
        evaluation = raw["evaluation"]
        dataset_root = _resolve(config_path.parent, dataset["root"])
        split_files = {
            name: _resolve(dataset_root, filename)
            for name, filename in dataset["split_files"].items()
        }
        config = ExperimentConfig(
            checkpoint_id=str(checkpoint["id"]),
            checkpoint_revision=checkpoint.get("revision"),
            checkpoint_cache_dir=(
                _resolve(config_path.parent, checkpoint["cache_dir"])
                if checkpoint.get("cache_dir")
                else None
            ),
            dataset_root=dataset_root,
            images_dir=_resolve(dataset_root, dataset["images_dir"]),
            captions_file=_resolve(dataset_root, dataset["captions_file"]),
            split_files=split_files,
            output_root=_resolve(config_path.parent, raw["output_root"]),
            expected_images_per_split=int(dataset["expected_images_per_split"]),
            captions_per_image=int(dataset["captions_per_image"]),
            crop_ratio=float(method["crop_ratio"]),
            crop_names=tuple(str(name) for name in method["crop_names"]),
            alpha_grid=tuple(float(value) for value in method["alpha_grid"]),
            seed=int(evaluation["seed"]),
            cutoffs=tuple(int(value) for value in evaluation["cutoffs"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid experiment configuration {config_path}: {exc}") from exc

    if not 0.0 < config.crop_ratio <= 1.0:
        raise ValueError("crop_ratio must be in the interval (0, 1]")
    if len(config.crop_names) != 5 or len(set(config.crop_names)) != 5:
        raise ValueError("crop_names must contain five unique names")
    if config.captions_per_image <= 0 or config.expected_images_per_split <= 0:
        raise ValueError("dataset counts must be positive")
    if not config.alpha_grid or any(value < 0.0 or value > 1.0 for value in config.alpha_grid):
        raise ValueError("alpha_grid values must be in the interval [0, 1]")
    if not config.cutoffs or any(value <= 0 for value in config.cutoffs):
        raise ValueError("evaluation cutoffs must be positive")
    return config

