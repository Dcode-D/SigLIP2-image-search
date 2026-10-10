"""Materialize the Flickr8k evaluation splits used by this project."""

from pathlib import Path
from typing import Mapping, Sequence


SPLIT_FILES = {
    "validation": "Flickr_8k.devImages.txt",
    "test": "Flickr_8k.testImages.txt",
}


def materialize_flickr8k(
    rows_by_split: Mapping[str, Sequence[Mapping]], target: Path
) -> dict:
    """Write Hugging Face Flickr8k rows in the project's standard file layout."""
    target = Path(target)
    images_dir = target / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    captions = []
    seen = set()
    summary = {}

    for split in ("validation", "test"):
        rows = rows_by_split.get(split)
        if rows is None:
            raise ValueError(f"missing {split} rows")
        image_ids = []
        for index, row in enumerate(rows):
            image = row.get("image") or {}
            image_id = Path(str(image.get("path") or f"{split}_{index:04d}.jpg")).name
            image_bytes = image.get("bytes")
            if not image_id or not image_bytes:
                raise ValueError(f"{split} row {index} has no image bytes")
            if image_id in seen:
                raise ValueError(f"duplicate image {image_id!r} across splits")
            seen.add(image_id)
            image_ids.append(image_id)
            (images_dir / image_id).write_bytes(image_bytes)

            for caption_index in range(5):
                caption = row.get(f"caption_{caption_index}")
                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError(
                        f"{split} row {index} has invalid caption_{caption_index}"
                    )
                captions.append(f"{image_id}#{caption_index}\t{caption.strip()}\n")

        (target / SPLIT_FILES[split]).write_text(
            "\n".join(image_ids) + "\n", encoding="utf-8"
        )
        summary[split] = len(image_ids)

    (target / "Flickr8k.token.txt").write_text("".join(captions), encoding="utf-8")
    return summary
