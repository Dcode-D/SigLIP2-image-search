import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from vision_retrieval.predictions import (
    PredictionRecord,
    read_predictions,
    validate_predictions,
    write_predictions,
)


class PredictionTestCase(unittest.TestCase):
    def setUp(self):
        self.gallery = ("a.jpg", "b.jpg", "c.jpg")
        self.expected_relevance = {
            "q1": ("a.jpg",),
            "q2": ("b.jpg",),
        }
        self.records = (
            PredictionRecord(
                query_id="q1",
                relevant_image_ids=("a.jpg",),
                ranked_image_ids=("a.jpg", "b.jpg", "c.jpg"),
                ranked_scores=(0.9, 0.8, 0.7),
                text_encoding_ms=2.0,
                search_ms=1.0,
                end_to_end_ms=3.0,
            ),
            PredictionRecord(
                query_id="q2",
                relevant_image_ids=("b.jpg",),
                ranked_image_ids=("c.jpg", "b.jpg", "a.jpg"),
                ranked_scores=(0.6, 0.5, 0.4),
                text_encoding_ms=2.5,
                search_ms=1.5,
                end_to_end_ms=4.0,
            ),
        )

    def test_prediction_round_trip_uses_gzip_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "predictions.jsonl.gz"
            write_predictions(path, self.records)
            loaded = read_predictions(path)

        self.assertEqual(loaded, self.records)

    def test_valid_predictions_are_accepted(self):
        validate_predictions(self.records, self.expected_relevance, self.gallery)

    def test_missing_or_duplicate_queries_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "query set"):
            validate_predictions(self.records[:1], self.expected_relevance, self.gallery)
        with self.assertRaisesRegex(ValueError, "duplicate query"):
            validate_predictions(
                (self.records[0], self.records[0]),
                {"q1": ("a.jpg",)},
                self.gallery,
            )

    def test_mismatched_relevance_is_rejected(self):
        changed = replace(self.records[0], relevant_image_ids=("b.jpg",))
        with self.assertRaisesRegex(ValueError, "relevance"):
            validate_predictions(
                (changed, self.records[1]), self.expected_relevance, self.gallery
            )

    def test_incomplete_or_duplicate_gallery_ranking_is_rejected(self):
        incomplete = replace(
            self.records[0],
            ranked_image_ids=("a.jpg", "b.jpg"),
            ranked_scores=(0.9, 0.8),
        )
        duplicate = replace(
            self.records[0],
            ranked_image_ids=("a.jpg", "a.jpg", "c.jpg"),
        )
        for record in (incomplete, duplicate):
            with self.subTest(record=record), self.assertRaisesRegex(ValueError, "gallery"):
                validate_predictions(
                    (record, self.records[1]), self.expected_relevance, self.gallery
                )

    def test_nonfinite_or_unsorted_scores_are_rejected(self):
        nonfinite = replace(self.records[0], ranked_scores=(0.9, math.nan, 0.7))
        unsorted = replace(self.records[0], ranked_scores=(0.9, 0.7, 0.8))
        for record in (nonfinite, unsorted):
            with self.subTest(record=record), self.assertRaisesRegex(ValueError, "scores"):
                validate_predictions(
                    (record, self.records[1]), self.expected_relevance, self.gallery
                )

    def test_negative_latency_is_rejected(self):
        changed = replace(self.records[0], search_ms=-1.0)
        with self.assertRaisesRegex(ValueError, "latency"):
            validate_predictions(
                (changed, self.records[1]), self.expected_relevance, self.gallery
            )


if __name__ == "__main__":
    unittest.main()
