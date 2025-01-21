import unittest
import numpy as np
import cv2
from typing import List, Tuple
from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat, draw_boxes_on_image, draw_boxes_on_image_with_labels_score
from saltup.utils.data.image.image_utils import generate_random_bgr_colors ,Image

class TestDrawBoxesOnImage(unittest.TestCase):
    def test_draw_boxes_on_image(self):
        """
        Test for the `draw_boxes_on_image` function.
        Verifies that bounding boxes are drawn correctly on the image.
        """
         # Create an empty image using the Image class (480x640, 3 color channels)
        image = Image(np.zeros((480, 640, 3), dtype=np.uint8))

        # Create some bounding boxes (in normalized format)
        bbox1 = BBox([0.2, 0.3, 0.4, 0.5], format=BBoxFormat.CENTER, img_width=640, img_height=480)
        bbox2 = BBox([0.6, 0.5, 0.3, 0.4], format=BBoxFormat.CENTER, img_width=640, img_height=480)

        # List of bounding boxes
        bboxes = [bbox1, bbox2]

        # Draw the bounding boxes on the image
        image_with_boxes = draw_boxes_on_image(image, bboxes, color=(0, 255, 0), thickness=2)

        # Verify that the image is not empty
        self.assertFalse(np.array_equal(image, image_with_boxes))

        # Verify that there are green pixels (color of the bounding boxes)
        green_pixels = np.where(
            (image_with_boxes.get_data()[:, :, 0] == 0) &
            (image_with_boxes.get_data()[:, :, 1] == 255) &
            (image_with_boxes.get_data()[:, :, 2] == 0)
        )
        self.assertGreater(len(green_pixels[0]), 0)  # There should be green pixels


class TestDrawBoxesOnImageWithLabelsScore(unittest.TestCase):
    def test_draw_text(self):
        # Create an empty image
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Draw white text
        cv2.putText(image, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Verify that there are white pixels
        white_pixels = np.where(
            (image[:, :, 0] == 255) &
            (image[:, :, 1] == 255) &
            (image[:, :, 2] == 255)
        )
        self.assertGreater(len(white_pixels[0]), 0)

    def test_draw_boxes_on_image_with_labels_score(self):
        """
        Test for the `draw_boxes_on_image_with_labels_score` function.
        Verifies that bounding boxes, labels, and scores are drawn correctly.
        """
        # Create an empty image using the Image class (480x640, 3 color channels)
        image = Image(np.zeros((480, 640, 3), dtype=np.uint8))

        # Create some bounding boxes (in normalized format)
        bbox1 = BBox([0.2, 0.3, 0.4, 0.5], format=BBoxFormat.CENTER, img_width=640, img_height=480)
        bbox2 = BBox([0.6, 0.5, 0.3, 0.4], format=BBoxFormat.CENTER, img_width=640, img_height=480)

        # List of tuples (BBox, class_id, score)
        bboxes_with_labels_score = [
            (bbox1, 1, 0.95),  # (BBox, class_id, score)
            (bbox2, 2, 0.87),
        ]

        # Determine the number of unique class IDs
        unique_class_ids = set(class_id for _, class_id, _ in bboxes_with_labels_score)
        num_classes = len(unique_class_ids)

        # Generate random colors for the classes
        class_colors = generate_random_bgr_colors(num_classes)  # Generate colors for all classes
        class_colors_dict = {class_id: color for class_id, color in zip(unique_class_ids, class_colors)}

        # Draw bounding boxes, labels, and scores on the image
        result = draw_boxes_on_image_with_labels_score(
            image,
            bboxes_with_labels_score,
            class_colors_bgr=class_colors_dict,
            thickness=2,  # Line thickness
            font_scale=0.5,  # Font scale
            text_color=(255, 255, 255),  # Text color (white)
            text_background_color=(0, 0, 0),  # Text background color (black)
        )

        # Ensure the result is a NumPy array (image)
        if isinstance(result, tuple):
            image_with_boxes = result[0]  # Extract the image from the tuple
        else:
            image_with_boxes = result

        # Verify that the image is not empty
        self.assertFalse(np.array_equal(image, image_with_boxes))

        # Verify that the bounding boxes are drawn with the correct colors
        for bbox, class_id, _ in bboxes_with_labels_score:
            # Get the color for the current class
            color = class_colors_dict[class_id]

            # Convert the bounding box coordinates to integers
            x1, y1, x2, y2 = bbox.get_coordinates(BBoxFormat.CORNERS)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Ensure coordinates are within the image dimensions
            x1 = max(0, min(x1, image_with_boxes.get_shape()[1] - 1))
            y1 = max(0, min(y1, image_with_boxes.get_shape()[0] - 1))
            x2 = max(0, min(x2, image_with_boxes.get_shape()[1] - 1))
            y2 = max(0, min(y2, image_with_boxes.get_shape()[0] - 1))

            # Check the color of the bounding box edges
            # Top edge
            top_edge = image_with_boxes.get_data()[y1, x1:x2, :]
            self.assertTrue(np.all(top_edge == color), f"Top edge color mismatch for class {class_id}")

            # Bottom edge
            bottom_edge = image_with_boxes.get_data()[y2, x1:x2, :]
            self.assertTrue(np.all(bottom_edge == color), f"Bottom edge color mismatch for class {class_id}")

            # Left edge
            left_edge = image_with_boxes.get_data()[y1:y2, x1, :]
            self.assertTrue(np.all(left_edge == color), f"Left edge color mismatch for class {class_id}")

            # Right edge
            right_edge = image_with_boxes.get_data()[y1:y2, x2, :]
            self.assertTrue(np.all(right_edge == color), f"Right edge color mismatch for class {class_id}")


if __name__ == "__main__":
    unittest.main()