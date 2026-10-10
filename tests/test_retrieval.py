import unittest

from vision_retrieval.retrieval import (
    FiveCropRetriever,
    GlobalRetriever,
    select_alpha,
)


class RetrievalTestCase(unittest.TestCase):
    def setUp(self):
        self.image_ids = ("b.jpg", "a.jpg", "c.jpg")
        self.global_embeddings = (
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
        )

    def test_global_retrieval_ranks_scores_and_breaks_ties_by_image_id(self):
        retriever = GlobalRetriever(self.image_ids, self.global_embeddings)

        results = retriever.search((1.0, 0.0))

        self.assertEqual([result.image_id for result in results], ["b.jpg", "c.jpg", "a.jpg"])
        self.assertEqual([result.score for result in results], [1.0, 1.0, 0.0])
        self.assertTrue(all(result.crop_score is None for result in results))
        self.assertTrue(all(result.winning_crop is None for result in results))

    def test_global_retrieval_supports_top_k(self):
        retriever = GlobalRetriever(self.image_ids, self.global_embeddings)

        results = retriever.search((0.0, 1.0), top_k=1)

        self.assertEqual([result.image_id for result in results], ["a.jpg"])

    def test_five_crop_retrieval_uses_maximum_crop_and_fusion(self):
        crop_names = ("tl", "tr", "bl", "br", "center")
        crop_embeddings = (
            ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            ((1.0, 0.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            ((0.5, 0.5), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        )
        retriever = FiveCropRetriever(
            self.image_ids,
            self.global_embeddings,
            crop_embeddings,
            crop_names,
            alpha=0.5,
        )

        results = retriever.search((1.0, 0.0))

        self.assertEqual([result.image_id for result in results], ["c.jpg", "a.jpg", "b.jpg"])
        by_id = {result.image_id: result for result in results}
        self.assertEqual(by_id["a.jpg"].global_score, 0.0)
        self.assertEqual(by_id["a.jpg"].crop_score, 1.0)
        self.assertEqual(by_id["a.jpg"].score, 0.5)
        self.assertEqual(by_id["a.jpg"].winning_crop, "tl")

    def test_invalid_retrieval_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            GlobalRetriever(("a.jpg",), ((1.0, 0.0), (0.0, 1.0)))
        with self.assertRaises(ValueError):
            FiveCropRetriever(
                ("a.jpg",),
                ((1.0, 0.0),),
                (((1.0, 0.0),) * 5,),
                ("tl", "tr", "bl", "br", "center"),
                alpha=1.1,
            )
        with self.assertRaises(ValueError):
            GlobalRetriever(("a.jpg",), ((1.0, 0.0),)).search((1.0,))
        with self.assertRaises(ValueError):
            GlobalRetriever(("a.jpg",), ((1.0, 0.0),)).search((1.0, 0.0), top_k=0)

    def test_select_alpha_uses_recall_then_mrr_then_smaller_alpha(self):
        image_ids = ("target.jpg", "other.jpg")
        global_embeddings = ((0.6, 0.8), (0.8, 0.6))
        crop_embeddings = (
            ((1.0, 0.0),) * 5,
            ((0.0, 1.0),) * 5,
        )
        query_vectors = ((1.0, 0.0), (0.0, 1.0))
        relevant_image_ids = ("target.jpg", "other.jpg")

        selection = select_alpha(
            query_vectors,
            relevant_image_ids,
            image_ids,
            global_embeddings,
            crop_embeddings,
            ("tl", "tr", "bl", "br", "center"),
            alpha_grid=(0.0, 0.5, 1.0),
        )

        self.assertEqual(selection.alpha, 0.5)
        self.assertEqual(selection.trials[0].recall_at_1, 0.0)
        self.assertEqual(selection.trials[1].recall_at_1, 1.0)
        self.assertEqual(selection.trials[2].recall_at_1, 1.0)

    def test_select_alpha_does_not_require_array_truthiness(self):
        class ArrayLike(list):
            def __bool__(self):
                raise ValueError("array truth value is ambiguous")

        selection = select_alpha(
            ArrayLike([(1.0, 0.0)]),
            ("target.jpg",),
            ("target.jpg",),
            ((1.0, 0.0),),
            (((1.0, 0.0),) * 5,),
            ("tl", "tr", "bl", "br", "center"),
            alpha_grid=(0.0,),
        )

        self.assertEqual(selection.alpha, 0.0)


if __name__ == "__main__":
    unittest.main()
