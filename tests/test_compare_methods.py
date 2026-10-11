"""Tests for scripts/compare_methods.py (Step 6 paired, image-clustered bootstrap)."""

import importlib.util
import math
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_methods.py"
spec = importlib.util.spec_from_file_location("compare_methods", SCRIPT)
compare_methods = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_methods)


def _predictions(rank_of_relevant: dict, gallery: list, relevance: dict) -> list:
    """Build rows where the relevant image sits at a chosen 1-based rank."""
    rows = []
    for query_id, rank in rank_of_relevant.items():
        others = [g for g in gallery if g != relevance[query_id]]
        ranked = others[: rank - 1] + [relevance[query_id]] + others[rank - 1 :]
        rows.append({"query_id": query_id, "ranked_image_ids": ranked})
    return rows


class PerQueryMetricsTest(unittest.TestCase):
    def test_rank_one_and_rank_three(self):
        gallery = [f"img{i}" for i in range(12)]
        relevance = {"q0": "img0", "q1": "img1"}
        rows = _predictions({"q0": 1, "q1": 3}, gallery, relevance)
        metrics = compare_methods.per_query_metrics(rows, relevance)
        self.assertEqual(metrics["q0"], {
            "recall_at_1": 1.0, "recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": 1.0, "ndcg_at_10": 1.0,
        })
        self.assertEqual(metrics["q1"]["recall_at_1"], 0.0)
        self.assertEqual(metrics["q1"]["recall_at_5"], 1.0)
        self.assertAlmostEqual(metrics["q1"]["mrr"], 1 / 3)
        self.assertAlmostEqual(metrics["q1"]["ndcg_at_10"], 1 / math.log2(4))

    def test_rank_outside_top_ten_scores_zero_ndcg(self):
        gallery = [f"img{i}" for i in range(12)]
        relevance = {"q0": "img0"}
        rows = _predictions({"q0": 11}, gallery, relevance)
        metrics = compare_methods.per_query_metrics(rows, relevance)
        self.assertEqual(metrics["q0"]["recall_at_10"], 0.0)
        self.assertEqual(metrics["q0"]["ndcg_at_10"], 0.0)
        self.assertAlmostEqual(metrics["q0"]["mrr"], 1 / 11)


class PairedBootstrapTest(unittest.TestCase):
    def setUp(self):
        self.gallery = [f"img{i}" for i in range(20)]
        # 4 images x 5 captions, as in Flickr8k
        self.relevance = {f"q{i}_{c}": f"img{i}" for i in range(4) for c in range(5)}
        self.image_of_query = dict(self.relevance)

    def _metrics(self, ranks: dict) -> dict:
        rows = _predictions(ranks, self.gallery, self.relevance)
        return compare_methods.per_query_metrics(rows, self.relevance)

    def test_identical_predictions_give_zero_interval(self):
        ranks = {q: 2 for q in self.relevance}
        base = self._metrics(ranks)
        report = compare_methods.paired_bootstrap(base, dict(base), self.image_of_query, samples=200, seed=1)
        for metric in compare_methods.METRICS:
            self.assertEqual(report[metric]["difference"], 0.0)
            self.assertEqual((report[metric]["ci95_low"], report[metric]["ci95_high"]), (0.0, 0.0))
            self.assertFalse(report[metric]["significant"])
        self.assertEqual(report["_meta"]["images"], 4)
        self.assertEqual(report["_meta"]["queries"], 20)

    def test_uniform_improvement_is_significant(self):
        base = self._metrics({q: 2 for q in self.relevance})
        cand = self._metrics({q: 1 for q in self.relevance})
        report = compare_methods.paired_bootstrap(base, cand, self.image_of_query, samples=200, seed=1)
        r1 = report["recall_at_1"]
        self.assertEqual(r1["difference"], 1.0)
        self.assertEqual((r1["ci95_low"], r1["ci95_high"]), (1.0, 1.0))
        self.assertTrue(r1["significant"])

    def test_mixed_result_interval_contains_zero(self):
        # candidate wins on images 0,1 and loses on images 2,3 by the same margin
        base_ranks = {q: (2 if q.startswith(("q0", "q1")) else 1) for q in self.relevance}
        cand_ranks = {q: (1 if q.startswith(("q0", "q1")) else 2) for q in self.relevance}
        report = compare_methods.paired_bootstrap(
            self._metrics(base_ranks), self._metrics(cand_ranks), self.image_of_query, samples=2000, seed=20261009
        )
        r1 = report["recall_at_1"]
        self.assertEqual(r1["difference"], 0.0)
        self.assertLess(r1["ci95_low"], 0.0)
        self.assertGreater(r1["ci95_high"], 0.0)
        self.assertFalse(r1["significant"])

    def test_seed_makes_result_deterministic(self):
        base = self._metrics({q: (3 if q.endswith("_0") else 1) for q in self.relevance})
        cand = self._metrics({q: (1 if q.endswith(("_0", "_1")) else 2) for q in self.relevance})
        a = compare_methods.paired_bootstrap(base, cand, self.image_of_query, samples=500, seed=7)
        b = compare_methods.paired_bootstrap(base, cand, self.image_of_query, samples=500, seed=7)
        self.assertEqual(a, b)

    def test_mismatched_query_sets_raise(self):
        base = self._metrics({q: 1 for q in self.relevance})
        cand = {k: v for k, v in base.items() if k != "q0_0"}
        with self.assertRaises(ValueError):
            compare_methods.paired_bootstrap(base, cand, self.image_of_query, samples=10, seed=1)


if __name__ == "__main__":
    unittest.main()
