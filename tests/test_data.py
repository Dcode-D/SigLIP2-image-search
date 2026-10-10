import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from vision_retrieval.config import load_config
from vision_retrieval.data import (
    load_split,
    manifest_sha256,
    validate_no_split_overlap,
    write_evaluation_manifests,
)


class ManifestTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset = self.root / "dataset"
        self.images = self.dataset / "images"
        self.images.mkdir(parents=True)

        self.image_ids = {
            "validation": ["val_a.jpg", "val_b.jpg"],
            "test": ["test_a.jpg", "test_b.jpg"],
        }
        for image_ids in self.image_ids.values():
            for image_id in image_ids:
                (self.images / image_id).write_bytes(b"fixture")

        captions = []
        for image_ids in self.image_ids.values():
            for image_id in image_ids:
                for caption_index in range(5):
                    captions.append(
                        f"{image_id}#{caption_index}\tcaption {caption_index} for {image_id}\n"
                    )
        (self.dataset / "Flickr8k.token.txt").write_text(
            "".join(captions), encoding="utf-8"
        )
        for split, image_ids in self.image_ids.items():
            (self.dataset / f"{split}.txt").write_text(
                "\n".join(image_ids) + "\n", encoding="utf-8"
            )

        self.config_path = self.root / "config.yaml"
        self.write_config()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_config(self, **overrides):
        config = {
            "checkpoint": {
                "id": "google/siglip2-base-patch16-224",
                "revision": "main",
                "cache_dir": "models",
            },
            "dataset": {
                "root": "dataset",
                "images_dir": "images",
                "captions_file": "Flickr8k.token.txt",
                "split_files": {
                    "validation": "validation.txt",
                    "test": "test.txt",
                },
                "expected_images_per_split": 2,
                "captions_per_image": 5,
            },
            "output_root": "artifacts",
            "method": {
                "crop_ratio": 0.6,
                "crop_names": [
                    "top_left",
                    "top_right",
                    "bottom_left",
                    "bottom_right",
                    "center",
                ],
                "alpha_grid": [0.0, 0.5, 1.0],
                "threshold_target_recall": 0.95,
            },
            "evaluation": {"seed": 20261009, "cutoffs": [1, 5, 10]},
        }
        for key, value in overrides.items():
            config[key] = value
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

    def test_load_split_creates_stable_records(self):
        config = load_config(self.config_path)

        images, queries = load_split(config, "validation")

        self.assertEqual(config.checkpoint_cache_dir, (self.root / "models").resolve())
        self.assertEqual([record.image_id for record in images], self.image_ids["validation"])
        self.assertEqual(len(queries), 10)
        self.assertEqual(queries[0].query_id, "validation:val_a.jpg:0")
        self.assertEqual(queries[-1].query_id, "validation:val_b.jpg:4")
        self.assertEqual(queries[0].relevant_image_id, "val_a.jpg")

    def test_load_split_requires_five_captions_per_image(self):
        captions_path = self.dataset / "Flickr8k.token.txt"
        lines = captions_path.read_text(encoding="utf-8").splitlines()
        captions_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "exactly 5 captions"):
            load_split(load_config(self.config_path), "test")

    def test_load_split_rejects_duplicate_caption_indices(self):
        captions_path = self.dataset / "Flickr8k.token.txt"
        with captions_path.open("a", encoding="utf-8") as handle:
            handle.write("test_a.jpg#0\tduplicate caption\n")

        with self.assertRaisesRegex(ValueError, "duplicate caption index"):
            load_split(load_config(self.config_path), "test")

    def test_load_split_rejects_duplicate_image_ids(self):
        split_path = self.dataset / "validation.txt"
        split_path.write_text("val_a.jpg\nval_a.jpg\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "duplicate image ID"):
            load_split(load_config(self.config_path), "validation")

    def test_load_split_reports_missing_image(self):
        (self.images / "test_b.jpg").unlink()

        with self.assertRaisesRegex(FileNotFoundError, "test_b.jpg"):
            load_split(load_config(self.config_path), "test")

    def test_validation_and_test_must_not_overlap(self):
        config = load_config(self.config_path)
        validation_images, _ = load_split(config, "validation")
        test_images, _ = load_split(config, "test")
        test_images[0] = test_images[0].with_image_id("val_a.jpg")

        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_no_split_overlap(
                {"validation": validation_images, "test": test_images}
            )

    def test_manifest_hash_is_deterministic_and_content_sensitive(self):
        config = load_config(self.config_path)
        images, queries = load_split(config, "validation")

        first = manifest_sha256(images, queries)
        second = manifest_sha256(images, queries)
        changed_queries = list(queries)
        changed_queries[0] = changed_queries[0].with_text("changed caption")

        self.assertEqual(first, second)
        self.assertNotEqual(first, manifest_sha256(images, changed_queries))

    def test_manifest_hash_ignores_local_storage_paths(self):
        config = load_config(self.config_path)
        images, queries = load_split(config, "validation")
        relocated = [
            replace(image, path=Path("/another-machine") / image.image_id)
            for image in images
        ]

        self.assertEqual(
            manifest_sha256(images, queries),
            manifest_sha256(relocated, queries),
        )

    def test_write_evaluation_manifests_writes_counts_and_hashes(self):
        config = load_config(self.config_path)

        metadata = write_evaluation_manifests(config)

        self.assertEqual(metadata["validation"]["image_count"], 2)
        self.assertEqual(metadata["validation"]["query_count"], 10)
        self.assertEqual(metadata["test"]["image_count"], 2)
        self.assertEqual(metadata["test"]["query_count"], 10)
        self.assertEqual(len(metadata["test"]["manifest_sha256"]), 64)
        self.assertTrue((config.output_root / "manifests" / "test_images.jsonl").is_file())
        self.assertTrue((config.output_root / "manifests" / "test_queries.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
