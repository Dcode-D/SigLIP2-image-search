import unittest

from vision_retrieval.demo import apply_score_threshold, threshold_status
from vision_retrieval.methods.retrievers import SearchResult


def _r(image_id, score):
    return SearchResult(image_id=image_id, score=score, global_score=score, crop_score=None, winning_crop=None)


class ApplyScoreThresholdTests(unittest.TestCase):
    def setUp(self):
        self.results = [_r("a", 0.30), _r("b", 0.12), _r("c", 0.1167), _r("d", 0.05)]

    def test_none_keeps_everything_with_ranks(self):
        kept, dropped = apply_score_threshold(self.results, None)
        self.assertEqual([rank for rank, _ in kept], [1, 2, 3, 4])
        self.assertEqual(dropped, 0)

    def test_cutoff_is_inclusive_and_keeps_original_ranks(self):
        kept, dropped = apply_score_threshold(self.results, 0.1167)
        self.assertEqual([(rank, r.image_id) for rank, r in kept], [(1, "a"), (2, "b"), (3, "c")])
        self.assertEqual(dropped, 1)

    def test_everything_below_cutoff(self):
        kept, dropped = apply_score_threshold(self.results, 0.5)
        self.assertEqual(kept, [])
        self.assertEqual(dropped, 4)


class ThresholdStatusTests(unittest.TestCase):
    def test_messages(self):
        self.assertIn("off", threshold_status("X", 0.1, "", 0.95, 3, 10, applied=False))
        self.assertIn("No validation-selected threshold", threshold_status("X", None, "", 0.95, 10, 10, True, "missing"))
        on = threshold_status("X", 0.1167, "global", 0.95, 3, 10, True)
        self.assertIn("0.1167", on); self.assertIn("`global`", on); self.assertIn("3 of the top 10", on)
        self.assertIn("No image", threshold_status("X", 0.1167, "", 0.95, 0, 10, True))


if __name__ == "__main__":
    unittest.main()
