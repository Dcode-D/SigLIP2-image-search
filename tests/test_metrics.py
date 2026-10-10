import math
import unittest

from vision_retrieval.metrics import (
    EvaluatedQuery,
    aggregate_metrics,
    evaluate_ranking,
    summarize_latencies,
)


class MetricsTestCase(unittest.TestCase):
    def test_rank_one_metrics(self):
        metrics = evaluate_ranking(("target", "a", "b"), {"target"})

        self.assertEqual(metrics.relevant_rank, 1)
        self.assertEqual(metrics.recall_at_1, 1.0)
        self.assertEqual(metrics.recall_at_5, 1.0)
        self.assertEqual(metrics.recall_at_10, 1.0)
        self.assertEqual(metrics.reciprocal_rank, 1.0)
        self.assertEqual(metrics.ndcg_at_10, 1.0)

    def test_rank_three_metrics(self):
        metrics = evaluate_ranking(("a", "b", "target", "c"), {"target"})

        self.assertEqual(metrics.relevant_rank, 3)
        self.assertEqual(metrics.recall_at_1, 0.0)
        self.assertEqual(metrics.recall_at_5, 1.0)
        self.assertEqual(metrics.reciprocal_rank, 1.0 / 3.0)
        self.assertEqual(metrics.ndcg_at_10, 0.5)

    def test_relevant_result_beyond_ten_has_zero_ndcg(self):
        ranked = tuple(f"image-{index}" for index in range(10)) + ("target",)

        metrics = evaluate_ranking(ranked, {"target"})

        self.assertEqual(metrics.relevant_rank, 11)
        self.assertEqual(metrics.recall_at_10, 0.0)
        self.assertEqual(metrics.ndcg_at_10, 0.0)
        self.assertEqual(metrics.reciprocal_rank, 1.0 / 11.0)

    def test_missing_relevant_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "relevant image"):
            evaluate_ranking(("a", "b"), {"target"})

    def test_latency_percentiles_use_linear_interpolation(self):
        summary = summarize_latencies((1.0, 2.0, 3.0, 4.0))

        self.assertEqual(summary.mean_ms, 2.5)
        self.assertEqual(summary.p50_ms, 2.5)
        self.assertAlmostEqual(summary.p95_ms, 3.85)

    def test_aggregate_metrics_averages_queries_and_latency(self):
        first = EvaluatedQuery(
            query_id="q1",
            metrics=evaluate_ranking(("target", "a"), {"target"}),
            text_encoding_ms=2.0,
            search_ms=1.0,
            end_to_end_ms=3.0,
        )
        second = EvaluatedQuery(
            query_id="q2",
            metrics=evaluate_ranking(("a", "target"), {"target"}),
            text_encoding_ms=4.0,
            search_ms=2.0,
            end_to_end_ms=6.0,
        )

        report = aggregate_metrics((first, second))

        self.assertEqual(report.query_count, 2)
        self.assertEqual(report.recall_at_1, 0.5)
        self.assertEqual(report.recall_at_5, 1.0)
        self.assertEqual(report.mrr, 0.75)
        self.assertEqual(report.text_encoding.mean_ms, 3.0)
        self.assertEqual(report.search.mean_ms, 1.5)
        self.assertEqual(report.end_to_end.mean_ms, 4.5)
        self.assertTrue(math.isfinite(report.ndcg_at_10))


if __name__ == "__main__":
    unittest.main()

