import tempfile
import unittest
from dataclasses import asdict, replace
import json
from pathlib import Path

from vision_retrieval.data.cache import (
    CacheMetadata,
    EmbeddingCache,
    assert_cache_compatible,
    load_embedding_cache,
    read_cache_metadata,
    save_embedding_cache,
)


class CacheTestCase(unittest.TestCase):
    def setUp(self):
        self.metadata = CacheMetadata(
            split="validation",
            checkpoint_id="google/siglip2-base-patch16-224",
            checkpoint_revision="abc123",
            manifest_sha256="1" * 64,
            crop_ratio=0.6,
            crop_names=(
                "top_left",
                "top_right",
                "bottom_left",
                "bottom_right",
                "center",
            ),
            image_ids=("a.jpg", "b.jpg"),
            vector_dimension=3,
            dtype="float32",
            normalized=True,
        )

    def test_compatible_metadata_is_accepted(self):
        assert_cache_compatible(self.metadata, self.metadata)

    def test_each_material_metadata_change_is_rejected(self):
        changes = {
            "checkpoint_revision": "different",
            "manifest_sha256": "2" * 64,
            "crop_ratio": 0.5,
            "crop_names": tuple(reversed(self.metadata.crop_names)),
            "image_ids": tuple(reversed(self.metadata.image_ids)),
            "vector_dimension": 4,
            "dtype": "float16",
            "normalized": False,
        }
        for field, value in changes.items():
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                assert_cache_compatible(self.metadata, replace(self.metadata, **{field: value}))

    def test_cache_metadata_can_be_read_without_loading_arrays(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            (path / "metadata.json").write_text(
                json.dumps(asdict(self.metadata)), encoding="utf-8"
            )

            loaded = read_cache_metadata(path)

        self.assertEqual(loaded, self.metadata)

    def test_numpy_cache_round_trip_when_numpy_is_available(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is not installed in the current runtime")

        global_embeddings = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        crop_embeddings = np.zeros((2, 5, 3), dtype=np.float32)
        crop_embeddings[:, :, 2] = 1.0
        cache = EmbeddingCache(
            metadata=self.metadata,
            global_embeddings=global_embeddings,
            crop_embeddings=crop_embeddings,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cache"
            save_embedding_cache(path, cache)
            loaded = load_embedding_cache(path, self.metadata)

        np.testing.assert_array_equal(loaded.global_embeddings, global_embeddings)
        np.testing.assert_array_equal(loaded.crop_embeddings, crop_embeddings)


if __name__ == "__main__":
    unittest.main()
