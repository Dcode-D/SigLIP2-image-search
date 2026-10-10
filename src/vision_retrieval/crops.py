"""Deterministic five-crop geometry for multi-view retrieval."""

from dataclasses import dataclass
from typing import Any, List, Tuple


CROP_NAMES = (
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
    "center",
)


@dataclass(frozen=True)
class CropRegion:
    name: str
    box: Tuple[int, int, int, int]


def crop_boxes(width: int, height: int, ratio: float = 0.6) -> List[CropRegion]:
    """Return five in-bounds `(left, top, right, bottom)` crop boxes."""
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("crop ratio must be in the interval (0, 1]")

    crop_width = min(width, max(1, round(width * ratio)))
    crop_height = min(height, max(1, round(height * ratio)))
    right_x = width - crop_width
    bottom_y = height - crop_height
    center_x = right_x // 2
    center_y = bottom_y // 2

    origins = (
        (0, 0),
        (right_x, 0),
        (0, bottom_y),
        (right_x, bottom_y),
        (center_x, center_y),
    )
    return [
        CropRegion(name, (x, y, x + crop_width, y + crop_height))
        for name, (x, y) in zip(CROP_NAMES, origins)
    ]


def five_crops(image: Any, ratio: float = 0.6) -> List[Tuple[str, Any]]:
    """Extract the five fixed crops from a Pillow-compatible image."""
    if not hasattr(image, "size") or not hasattr(image, "crop"):
        raise TypeError("image must provide Pillow-compatible size and crop attributes")
    width, height = image.size
    return [(region.name, image.crop(region.box)) for region in crop_boxes(width, height, ratio)]

