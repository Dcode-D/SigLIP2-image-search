import unittest

import numpy as np

from vision_retrieval.methods.faiss_retrievers import FaissRetriever, complete_ranking
from vision_retrieval.methods.retrievers import GlobalRetriever


def _gallery(n=300, d=32, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    ids = tuple(f"img_{i:04d}.jpg" for i in range(n))
    return ids, x


def _queries(d=32, count=20, seed=11):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((count, d)).astype(np.float32)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


class CompleteRankingTests(unittest.TestCase):
    def test_fills_missing_ids_below_every_returned_score_in_id_order(self):
        ids = ("c.jpg", "a.jpg", "b.jpg", "d.jpg")
        ranked, scores = complete_ranking(ids, [2, -1, 0, 7], [0.9, 0.0, 0.5, 0.1])
        self.assertEqual(ranked, ["b.jpg", "c.jpg", "a.jpg", "d.jpg"])
        self.assertEqual(scores[:2], [0.9, 0.5])
        self.assertTrue(all(s < 0.5 for s in scores[2:]))
        self.assertEqual(scores[2], scores[3])  # fillers tie, ordered by id

    def test_drops_duplicates_and_keeps_descending_order(self):
        ids = ("a", "b", "c")
        ranked, scores = complete_ranking(ids, [1, 1, 0], [0.8, 0.8, 0.9])
        self.assertEqual(ranked, ["a", "b", "c"])
        self.assertEqual(len(set(ranked)), 3)
        self.assertTrue(all(x >= y for x, y in zip(scores, scores[1:])))


class FaissRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.ids, self.x = _gallery()
        self.q = _queries()
        self.exact = GlobalRetriever(self.ids, self.x)

    def _exact_ids(self, q):
        return [r.image_id for r in self.exact.search(q)]

    def _top10_overlap(self, retriever, q):
        got = {r.image_id for r in retriever.search(q, top_k=10)}
        return len(set(self._exact_ids(q)[:10]) & got) / 10

    def test_flat_matches_exact_numpy_ranking(self):
        flat = FaissRetriever(self.ids, self.x, "flat")
        for q in self.q:
            got = flat.search(q)
            self.assertEqual([r.image_id for r in got], self._exact_ids(q))
            np.testing.assert_allclose(
                [r.score for r in got], [r.score for r in self.exact.search(q)], atol=1e-5
            )

    def test_ivf_returns_a_valid_full_ranking_even_when_cells_are_missed(self):
        # nprobe=1 of 8 cells: most of the gallery is unreachable and must be filled in
        retriever = FaissRetriever(self.ids, self.x, "ivf", {"nlist": 8, "nprobe": 1})
        results = retriever.search(self.q[0])
        ids = [r.image_id for r in results]
        scores = [r.score for r in results]
        self.assertEqual(len(ids), len(self.ids))
        self.assertEqual(set(ids), set(self.ids))
        self.assertTrue(all(a >= b for a, b in zip(scores, scores[1:])))
        for (ia, sa), (ib, sb) in zip(zip(ids, scores), zip(ids[1:], scores[1:])):
            if sa == sb:
                self.assertLess(ia, ib)
        self.assertGreater(retriever.info.index_bytes, 0)
        self.assertGreaterEqual(retriever.info.build_seconds, 0.0)

    def test_top_k_truncates(self):
        ivf = FaissRetriever(self.ids, self.x, "ivf", {"nlist": 8, "nprobe": 2})
        self.assertEqual(len(ivf.search(self.q[0], top_k=5)), 5)

    def test_ivf_with_all_cells_probed_equals_exact(self):
        ivf = FaissRetriever(self.ids, self.x, "ivf", {"nlist": 8, "nprobe": 8})
        for q in self.q[:5]:
            self.assertEqual([r.image_id for r in ivf.search(q)][:10], self._exact_ids(q)[:10])

    def test_ivf_recall_grows_with_nprobe(self):
        # visiting more cells only adds candidates, so top-10 overlap cannot decrease
        retrievers = [
            FaissRetriever(self.ids, self.x, "ivf", {"nlist": 8, "nprobe": nprobe}) for nprobe in (1, 2, 4, 8)
        ]
        for q in self.q:
            overlaps = [self._top10_overlap(r, q) for r in retrievers]
            self.assertEqual(overlaps, sorted(overlaps))
            self.assertEqual(overlaps[-1], 1.0)

    def test_ivf_scores_are_cosines_like_the_baseline(self):
        ivf = FaissRetriever(self.ids, self.x, "ivf", {"nlist": 8, "nprobe": 8})
        q = self.q[0]
        expected = {r.image_id: r.score for r in self.exact.search(q)}
        for r in ivf.search(q, top_k=10):
            self.assertAlmostEqual(r.score, expected[r.image_id], places=5)

    def test_rejects_unnormalised_or_bad_input(self):
        with self.assertRaises(ValueError):
            FaissRetriever(self.ids, self.x * 2.0, "flat")
        with self.assertRaises(ValueError):
            FaissRetriever(self.ids, self.x, "ivf", {"nlist": 4, "nprobe": 9})
        with self.assertRaises(ValueError):
            FaissRetriever(self.ids, self.x, "ivf", {"nlist": 0, "nprobe": 1})
        with self.assertRaises(ValueError):
            FaissRetriever(self.ids, self.x, "hnsw")

    def test_builds_are_deterministic_for_same_seed(self):
        a = FaissRetriever(self.ids, self.x, "ivf", {"nlist": 8, "nprobe": 2}, seed=3)
        b = FaissRetriever(self.ids, self.x, "ivf", {"nlist": 8, "nprobe": 2}, seed=3)
        for q in self.q[:5]:
            self.assertEqual([r.image_id for r in a.search(q)], [r.image_id for r in b.search(q)])


if __name__ == "__main__":
    unittest.main()
