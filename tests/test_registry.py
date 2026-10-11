import unittest
from unittest import mock

import numpy as np

from vision_retrieval.methods import registry
from vision_retrieval.data.cache import CacheMetadata, EmbeddingCache
from vision_retrieval.methods import (
    BuildContext,
    MethodSpec,
    build_method,
    get_method,
    list_methods,
    method_labels,
    register,
)


def _fake_cache(n=50, d=16):
    rng = np.random.default_rng(0)
    g = rng.standard_normal((n, d)).astype(np.float32)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    c = rng.standard_normal((n, 5, d)).astype(np.float32)
    c /= np.linalg.norm(c, axis=-1, keepdims=True)
    ids = tuple(f"{i:03d}.jpg" for i in range(n))
    meta = CacheMetadata(
        split="test",
        checkpoint_id="ckpt",
        checkpoint_revision="rev",
        manifest_sha256="hash",
        crop_ratio=0.6,
        crop_names=("top_left", "top_right", "bottom_left", "bottom_right", "center"),
        image_ids=ids,
        vector_dimension=d,
    )
    return EmbeddingCache(meta, g, c)


class RegistryTests(unittest.TestCase):
    def test_builtin_methods_are_registered_in_order(self):
        names = list_methods()
        self.assertEqual(names[:2], ("global", "five_crop"))
        for expected in ("faiss_flat", "faiss_ivf"):
            self.assertIn(expected, names)
        self.assertEqual(len(names), 4)

    def test_labels_are_unique_and_map_back_to_names(self):
        labels = method_labels()
        self.assertEqual(len(labels), len(list_methods()))
        for label, name in labels.items():
            self.assertEqual(get_method(name).label, label)

    def test_unknown_method_raises_with_hint(self):
        with self.assertRaises(KeyError) as ctx:
            get_method("does_not_exist")
        self.assertIn("global", str(ctx.exception))

    def test_duplicate_registration_is_rejected(self):
        spec = get_method("global")
        with self.assertRaises(ValueError):
            register(spec)

    def test_register_and_build_custom_method(self):
        calls = []

        def builder(context):
            calls.append(context.cache.metadata.split)
            return object(), {"k": 1}

        spec = MethodSpec("custom_test", "Custom test", "test", "desc", builder, {"base": 0})
        register(spec)
        try:
            cache = _fake_cache()
            built = build_method("custom_test", BuildContext(mock.Mock(), cache, "rev"))
            self.assertEqual(calls, ["test"])
            self.assertEqual(built.parameters, {"base": 0, "k": 1})
            self.assertGreaterEqual(built.build_seconds, 0.0)
        finally:
            registry._REGISTRY.pop("custom_test", None)

    def test_global_and_faiss_methods_build_from_cache_without_files(self):
        cache = _fake_cache()
        context = BuildContext(mock.Mock(), cache, "rev")
        q = cache.global_embeddings[0]
        exact = build_method("global", context).retriever.search(q)
        for name in ("faiss_flat", "faiss_ivf"):
            with self.subTest(method=name):
                built = build_method(name, context)
                results = built.retriever.search(q)
                self.assertEqual(len(results), len(cache.metadata.image_ids))
                self.assertIn("index_bytes", built.parameters)
                if name == "faiss_flat":
                    self.assertEqual([r.image_id for r in results], [r.image_id for r in exact])


if __name__ == "__main__":
    unittest.main()
