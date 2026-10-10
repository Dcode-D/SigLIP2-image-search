import unittest

from vision_retrieval.crops import CROP_NAMES, crop_boxes, five_crops


class CropGeometryTestCase(unittest.TestCase):
    def test_landscape_crop_boxes(self):
        regions = crop_boxes(10, 8)

        self.assertEqual([region.name for region in regions], list(CROP_NAMES))
        self.assertEqual(
            [region.box for region in regions],
            [
                (0, 0, 6, 5),
                (4, 0, 10, 5),
                (0, 3, 6, 8),
                (4, 3, 10, 8),
                (2, 1, 8, 6),
            ],
        )

    def test_portrait_crop_boxes(self):
        self.assertEqual(
            [region.box for region in crop_boxes(8, 10)],
            [
                (0, 0, 5, 6),
                (3, 0, 8, 6),
                (0, 4, 5, 10),
                (3, 4, 8, 10),
                (1, 2, 6, 8),
            ],
        )

    def test_square_and_one_pixel_images_have_valid_boxes(self):
        for width, height in ((10, 10), (1, 1)):
            for region in crop_boxes(width, height):
                left, top, right, bottom = region.box
                self.assertGreater(right, left)
                self.assertGreater(bottom, top)
                self.assertGreaterEqual(left, 0)
                self.assertGreaterEqual(top, 0)
                self.assertLessEqual(right, width)
                self.assertLessEqual(bottom, height)

    def test_invalid_dimensions_and_ratios_are_rejected(self):
        for args in ((0, 10, 0.6), (10, 0, 0.6), (10, 10, 0.0), (10, 10, 1.1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                crop_boxes(*args)

    def test_five_crops_uses_pillow_when_available(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed in the current runtime")

        image = Image.new("RGB", (10, 8))
        crops = five_crops(image)

        self.assertEqual([name for name, _ in crops], list(CROP_NAMES))
        self.assertEqual([crop.size for _, crop in crops], [(6, 5)] * 5)


if __name__ == "__main__":
    unittest.main()

