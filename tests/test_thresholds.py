import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from vision_retrieval.evaluation.metrics import select_score_threshold
from vision_retrieval.methods import get_method
from vision_retrieval.evaluation.predictions import PredictionRecord
from vision_retrieval.evaluation.thresholds import (
    load_threshold,
    reachable_fraction,
    score_matrix_from_records,
    select_threshold_from_records,
    threshold_file_contents,
    threshold_path,
    write_threshold_file,
)

GALLERY = ("a", "b", "c", "d")


def _record(query_id, relevant, ranked, scores):
    return PredictionRecord(query_id, (relevant,), tuple(ranked), tuple(scores), 1.0, 1.0, 2.0)


class ScoreMatrixTests(unittest.TestCase):
    def test_matrix_is_in_gallery_order_with_relevant_columns(self):
        records = [
            _record("q1", "c", ["c", "a", "d", "b"], [0.9, 0.5, 0.2, 0.1]),
            _record("q2", "a", ["b", "a", "c", "d"], [0.8, 0.7, 0.3, 0.0]),
        ]
        matrix, relevant = score_matrix_from_records(GALLERY, records)
        np.testing.assert_allclose(matrix[0], [0.5, 0.1, 0.9, 0.2])
        np.testing.assert_allclose(matrix[1], [0.7, 0.8, 0.3, 0.0])
        self.assertEqual(relevant.tolist(), [2, 0])

    def test_selection_from_records_matches_direct_selection(self):
        rng = np.random.default_rng(0)
        gallery = tuple(f"img{i}" for i in range(50))
        records = []
        for q in range(40):
            scores = rng.uniform(-0.2, 0.4, size=50).astype(np.float32)
            order = np.argsort(-scores)
            records.append(_record(f"q{q}", gallery[q], [gallery[i] for i in order], scores[order].tolist()))
        selection = select_threshold_from_records(records, gallery, 0.9)
        matrix, relevant = score_matrix_from_records(gallery, records)
        direct = select_score_threshold(matrix, relevant, 0.9)
        self.assertEqual(selection, direct)
        self.assertGreaterEqual(selection.recall, 0.9)

    def test_rejects_partial_or_foreign_rankings(self):
        with self.assertRaises(ValueError):
            score_matrix_from_records(GALLERY, [_record("q", "a", ["a", "b"], [0.2, 0.1])])
        with self.assertRaises(ValueError):
            score_matrix_from_records(GALLERY, [_record("q", "a", ["a", "b", "c", "zzz"], [0.4, 0.3, 0.2, 0.1])])
        with self.assertRaises(ValueError):
            score_matrix_from_records(GALLERY, [_record("q", "a", ["a", "a", "c", "d"], [0.4, 0.3, 0.2, 0.1])])


class ReachableFractionTests(unittest.TestCase):
    def test_filler_scored_relevant_counts_as_unreachable(self):
        records = [
            _record("q1", "a", ["a", "b", "c", "d"], [0.5, 0.4, -0.6, -0.6]),  # reachable
            _record("q2", "c", ["a", "b", "c", "d"], [0.5, 0.4, -0.6, -0.6]),  # c sits at the filler level
        ]
        self.assertEqual(reachable_fraction(records), 0.5)


class ThresholdFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "global_threshold.json"
        selection = select_score_threshold(np.array([[0.3, 0.1], [0.2, 0.4]], dtype=np.float32), np.array([0, 1]), 0.95)
        contents = threshold_file_contents(
            selection,
            method_name="global",
            method_parameters={"alpha": 0.0, "index_parameters": {"nlist": 32}},
            checkpoint_id="ckpt",
            checkpoint_revision="rev1",
            validation_manifest_sha256="hash1",
            query_count=2,
            gallery_size=2,
        )
        write_threshold_file(self.path, contents)
        self.contents = contents

    def tearDown(self):
        self.tmp.cleanup()

    def test_contents_carry_selection_and_provenance(self):
        data = json.loads(self.path.read_text())
        self.assertEqual(data["method_name"], "global")
        self.assertEqual(data["checkpoint_revision"], "rev1")
        self.assertEqual(data["target_recall"], 0.95)
        self.assertEqual(data["method_parameters"]["index_parameters"]["nlist"], 32)
        self.assertAlmostEqual(data["threshold"], 0.3, places=6)  # positives 0.3, 0.4; keep both

    def test_load_checks_provenance(self):
        ok = load_threshold(self.path, validation_manifest_sha256="hash1", checkpoint_revision="rev1", target_recall=0.95, alpha=0.0)
        self.assertAlmostEqual(ok, 0.3, places=6)
        with self.assertRaises(ValueError):
            load_threshold(self.path, validation_manifest_sha256="other", checkpoint_revision="rev1", target_recall=0.95)
        with self.assertRaises(ValueError):
            load_threshold(self.path, validation_manifest_sha256="hash1", checkpoint_revision="rev2", target_recall=0.95)
        with self.assertRaises(ValueError):
            load_threshold(self.path, validation_manifest_sha256="hash1", checkpoint_revision="rev1", target_recall=0.9)
        with self.assertRaises(ValueError):
            load_threshold(self.path, validation_manifest_sha256="hash1", checkpoint_revision="rev1", target_recall=0.95, alpha=0.5)
        with self.assertRaises(FileNotFoundError):
            load_threshold(self.path.with_name("missing_threshold.json"), validation_manifest_sha256="hash1", checkpoint_revision="rev1", target_recall=0.95)


class ThresholdLinkTests(unittest.TestCase):
    def test_registry_links(self):
        self.assertEqual(get_method("global").threshold_method, "global")
        self.assertEqual(get_method("five_crop").threshold_method, "five_crop")
        self.assertEqual(get_method("faiss_flat").threshold_method, "global")
        self.assertEqual(get_method("faiss_ivf").threshold_method, "global")

    def test_threshold_path_follows_the_link(self):
        config = mock.Mock(output_root=Path("/tmp/artifacts"))
        self.assertEqual(threshold_path(config, get_method("faiss_ivf")), Path("/tmp/artifacts/parameters/global_threshold.json"))
        self.assertEqual(threshold_path(config, get_method("five_crop")), Path("/tmp/artifacts/parameters/five_crop_threshold.json"))


if __name__ == "__main__":
    unittest.main()
