import sys
import os
import unittest
import torch
from PIL import Image

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from gdino_module.gdino import GroundingDINO
from gdino_module.utils import calculate_iou


class TestGroundingDINO(unittest.TestCase):
    def setUp(self):
        self.gdino = GroundingDINO()

        self.gdino_overlay_bbox_result_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "gdino_overlay_images"
        )
        if not os.path.exists(self.gdino_overlay_bbox_result_dir):
            os.makedirs(self.gdino_overlay_bbox_result_dir)

        self.local_test_images = [
            {
                "image_id": "table0.png",
                "expected_labels": ["an apple"],
                "expected_boxes": torch.tensor(
                    [[871.3726, 606.1895, 1021.9732, 763.6394]], device="cuda:0"
                ),
                "text": "an apple. a orange",
            },
            {
                "image_id": "table1.png",
                "expected_labels": ["a box", "a can", "a box"],
                "expected_boxes": torch.tensor(
                    [
                        [305.2202, 539.2209, 579.2434, 914.7086],
                        [1335.8514, 363.2420, 1594.6263, 782.9238],
                        [884.4767, 498.0680, 1114.0472, 798.0426],
                    ],
                    device="cuda:0",
                ),
                "text": "a box of Oreo. a can of Monster. a can of Pringles",
            },
            {
                "image_id": "table2.png",
                "expected_labels": ["a red bull", "a can"],
                "expected_boxes": torch.tensor(
                    [
                        [922.7785, 976.9618, 1118.4335, 1346.5343],
                        [169.4472, 873.1445, 402.1467, 1318.3616],
                    ],
                    device="cuda:0",
                ),
                "text": "a Red Bull can. a can of Pringles. a lemon",
            },
        ]

    def test_load_image_from_url(self):
        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = self.gdino.load_image_from_url(image_url)
        self.assertIsInstance(image, Image.Image)

    def test_detect_objects(self):
        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = self.gdino.load_image_from_url(image_url)
        text = "a cat. a remote control."
        results = self.gdino.detect_objects(image, text, clean_output=False)
        self.assertTrue(len(results) > 0)

    def test_local_image_with_bboxes(self):
        for test_image in self.local_test_images:
            image_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "images",
                test_image["image_id"],
            )
            image = Image.open(image_path).convert("RGB")
            text = test_image["text"]

            results = self.gdino.detect_objects(image, text, clean_output=False)
            boxes = results[0]["boxes"]
            labels = results[0]["labels"]
            scores = results[0]["scores"]

            # Assert that the detected objects match the expected results
            expected_labels = test_image["expected_labels"]
            expected_boxes = test_image["expected_boxes"]

            for expected_label, actual_label in zip(expected_labels, labels):
                self.assertIn(expected_label, actual_label)

            # IoU
            for expected_box, actual_box in zip(expected_boxes, boxes):
                iou = calculate_iou(expected_box.tolist(), actual_box.tolist())
                self.assertGreater(iou, 0.65)

            image_with_boxes = self.gdino.draw_boxes(image.copy(), boxes, labels)

            output_image_path = os.path.join(
                self.gdino_overlay_bbox_result_dir, test_image["image_id"]
            )
            image_with_boxes.save(output_image_path)

            self.assertTrue(len(boxes) > 0)


if __name__ == "__main__":
    unittest.main()

    # pytest -s tests/test_gdino.py
