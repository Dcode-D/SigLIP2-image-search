import tempfile
import unittest
from pathlib import Path

from vision_retrieval.data.setup_assets import materialize_flickr8k


class SetupAssetsTestCase(unittest.TestCase):
    def test_materializes_only_validation_and_test_in_legacy_layout(self):
        rows = {
            "validation": [
                {
                    "image": {"path": "val.jpg", "bytes": b"validation-image"},
                    **{f"caption_{index}": f"validation caption {index}" for index in range(5)},
                }
            ],
            "test": [
                {
                    "image": {"path": "test.jpg", "bytes": b"test-image"},
                    **{f"caption_{index}": f"test caption {index}" for index in range(5)},
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary = materialize_flickr8k(rows, root)

            self.assertEqual(summary, {"validation": 1, "test": 1})
            self.assertEqual(
                (root / "Flickr_8k.devImages.txt").read_text(encoding="utf-8"),
                "val.jpg\n",
            )
            self.assertEqual(
                (root / "Flickr_8k.testImages.txt").read_text(encoding="utf-8"),
                "test.jpg\n",
            )
            self.assertEqual((root / "images" / "val.jpg").read_bytes(), b"validation-image")
            captions = (root / "Flickr8k.token.txt").read_text(encoding="utf-8")
            self.assertIn("val.jpg#0\tvalidation caption 0\n", captions)
            self.assertIn("test.jpg#4\ttest caption 4\n", captions)

    def test_rejects_duplicate_image_names_across_splits(self):
        row = {
            "image": {"path": "same.jpg", "bytes": b"image"},
            **{f"caption_{index}": f"caption {index}" for index in range(5)},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "duplicate image"):
                materialize_flickr8k(
                    {"validation": [row], "test": [row]}, Path(temp_dir)
                )


if __name__ == "__main__":
    unittest.main()
