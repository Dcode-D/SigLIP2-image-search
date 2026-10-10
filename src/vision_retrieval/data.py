"""Flickr8k split loading and canonical manifest generation."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from .config import ExperimentConfig


@dataclass(frozen=True)
class ImageRecord:
    split: str
    image_id: str
    path: Path

    def with_image_id(self, image_id: str) -> "ImageRecord":
        return replace(self, image_id=image_id)


@dataclass(frozen=True)
class QueryRecord:
    split: str
    query_id: str
    relevant_image_id: str
    caption_index: int
    text: str

    def with_text(self, text: str) -> "QueryRecord":
        return replace(self, text=text)


def _read_split_ids(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"split file not found: {path}")
    image_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = set()
    for image_id in image_ids:
        if image_id in seen:
            raise ValueError(f"duplicate image ID {image_id!r} in {path}")
        seen.add(image_id)
    return image_ids


def _read_captions(path: Path) -> Dict[str, Dict[int, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"caption file not found: {path}")
    captions = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            key, caption = raw_line.split("\t", 1)
            image_id, index_text = key.rsplit("#", 1)
            caption_index = int(index_text)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid caption line {line_number} in {path}") from exc
        if not caption.strip():
            raise ValueError(f"empty caption at line {line_number} in {path}")
        image_captions = captions.setdefault(image_id, {})
        if caption_index in image_captions:
            raise ValueError(
                f"duplicate caption index {caption_index} for {image_id} in {path}"
            )
        image_captions[caption_index] = caption.strip()
    return captions


def load_split(
    config: ExperimentConfig, split: str
) -> Tuple[List[ImageRecord], List[QueryRecord]]:
    """Load and validate one Flickr8k evaluation split."""
    if split not in config.split_files:
        raise ValueError(f"unknown split {split!r}; configured splits: {sorted(config.split_files)}")

    image_ids = _read_split_ids(config.split_files[split])
    if len(image_ids) != config.expected_images_per_split:
        raise ValueError(
            f"{split} must contain exactly {config.expected_images_per_split} images; "
            f"found {len(image_ids)}"
        )
    captions_by_image = _read_captions(config.captions_file)
    expected_indices = set(range(config.captions_per_image))
    images = []
    queries = []

    for image_id in image_ids:
        image_path = config.images_dir / image_id
        if not image_path.is_file():
            raise FileNotFoundError(f"image {image_id!r} not found at {image_path}")
        captions = captions_by_image.get(image_id, {})
        if set(captions) != expected_indices:
            raise ValueError(
                f"{image_id} must have exactly {config.captions_per_image} captions "
                f"with indices {sorted(expected_indices)}; found {sorted(captions)}"
            )

        images.append(ImageRecord(split=split, image_id=image_id, path=image_path.resolve()))
        for caption_index in sorted(captions):
            queries.append(
                QueryRecord(
                    split=split,
                    query_id=f"{split}:{image_id}:{caption_index}",
                    relevant_image_id=image_id,
                    caption_index=caption_index,
                    text=captions[caption_index],
                )
            )
    return images, queries


def validate_no_split_overlap(split_images: Mapping[str, Sequence[ImageRecord]]) -> None:
    """Reject any image ID appearing in more than one configured split."""
    owners = {}
    for split, images in split_images.items():
        for image in images:
            previous = owners.get(image.image_id)
            if previous is not None:
                raise ValueError(
                    f"split overlap: image {image.image_id!r} appears in {previous} and {split}"
                )
            owners[image.image_id] = split


def _image_dict(record: ImageRecord) -> dict:
    values = asdict(record)
    values["path"] = str(record.path)
    return values


def _query_dict(record: QueryRecord) -> dict:
    return asdict(record)


def manifest_sha256(
    images: Sequence[ImageRecord], queries: Sequence[QueryRecord]
) -> str:
    """Hash canonical manifest content in its ranking-relevant order."""
    payload = {
        "images": [
            {"split": record.split, "image_id": record.image_id}
            for record in images
        ],
        "queries": [_query_dict(record) for record in queries],
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_evaluation_manifests(config: ExperimentConfig) -> dict:
    """Create canonical validation/test manifests and return their metadata."""
    loaded = {split: load_split(config, split) for split in ("validation", "test")}
    validate_no_split_overlap({split: records[0] for split, records in loaded.items()})

    manifest_dir = config.output_root / "manifests"
    metadata = {}
    for split, (images, queries) in loaded.items():
        _write_jsonl(manifest_dir / f"{split}_images.jsonl", [_image_dict(row) for row in images])
        _write_jsonl(manifest_dir / f"{split}_queries.jsonl", [_query_dict(row) for row in queries])
        metadata[split] = {
            "image_count": len(images),
            "query_count": len(queries),
            "manifest_sha256": manifest_sha256(images, queries),
        }

    metadata_path = manifest_dir / "metadata.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(metadata_path)
    return metadata

