import unittest
import numpy as np
import cv2
from typing import List, Tuple


from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat , draw_boxes_on_image, draw_boxes_on_image_with_labels_score

class TestDrawBoxesOnImage(unittest.TestCase):
    def test_draw_boxes_on_image(self):
        """
        Test per la funzione `draw_boxes_on_image`.
        Verifica che le bounding box vengano disegnate correttamente sull'immagine.
        """
        # Crea un'immagine vuota (480x640, 3 canali di colore)
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Crea alcune bounding box (in formato normalizzato)
        bbox1 = BBox([0.2, 0.3, 0.4, 0.5], format=BBoxFormat.CENTER, img_width=640, img_height=480)
        bbox2 = BBox([0.6, 0.5, 0.3, 0.4], format=BBoxFormat.CENTER, img_width=640, img_height=480)

        # Lista di bounding box
        bboxes = [bbox1, bbox2]

        # Disegna le bounding box sull'immagine
        image_with_boxes = draw_boxes_on_image(image, bboxes, color=(0, 255, 0), thickness=2)

        # Verifica che l'immagine non sia vuota
        self.assertFalse(np.array_equal(image, image_with_boxes))

        # Verifica che ci siano pixel verdi (colore delle bounding box)
        green_pixels = np.where(
            (image_with_boxes[:, :, 0] == 0) &
            (image_with_boxes[:, :, 1] == 255) &
            (image_with_boxes[:, :, 2] == 0)
        )
        self.assertGreater(len(green_pixels[0]), 0)  # Ci dovrebbero essere pixel verdi

class TestDrawBoxesOnImageWithLabelsScore(unittest.TestCase):
  

    def test_draw_text(self):
        # Crea un'immagine vuota
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Disegna un testo bianco
        cv2.putText(image, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Verifica che ci siano pixel bianchi
        white_pixels = np.where(
            (image[:, :, 0] == 255) &
            (image[:, :, 1] == 255) &
            (image[:, :, 2] == 255)
        )
        self.assertGreater(len(white_pixels[0]), 0)

    def test_draw_boxes_on_image_with_labels_score(self):
        """
        Test per la funzione `draw_boxes_on_image_with_labels_score`.
        Verifica che le bounding box, le etichette e i punteggi vengano disegnati correttamente.
        """
        # Crea un'immagine vuota (480x640, 3 canali di colore)
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Crea alcune bounding box (in formato normalizzato)
        bbox1 = BBox([0.2, 0.3, 0.4, 0.5], format=BBoxFormat.CENTER, img_width=640, img_height=480)
        bbox2 = BBox([0.6, 0.5, 0.3, 0.4], format=BBoxFormat.CENTER, img_width=640, img_height=480)

        # Lista di tuple (BBox, class_id, score)
        bboxes_with_labels_score = [
            (bbox1, 1, 0.95),  # (BBox, class_id, score)
            (bbox2, 2, 0.87),
        ]

        # Disegna le bounding box con etichette e punteggi sull'immagine
        image_with_boxes = draw_boxes_on_image_with_labels_score(
            image,
            bboxes_with_labels_score,
            color=(0, 255, 0),  # Colore delle box (verde)
            thickness=10,        # Spessore delle linee
            font_scale=1,     # Scala del font
            text_color=(255, 255, 255),  # Colore del testo (bianco)
            text_background_color=(0, 0, 0),  # Colore dello sfondo del testo (nero)
        )

        # Verifica che l'immagine non sia vuota
        self.assertFalse(np.array_equal(image, image_with_boxes))

        # Verifica che ci siano pixel verdi (colore delle bounding box)
        green_pixels = np.where(
            (image_with_boxes[:, :, 0] == 0) &
            (image_with_boxes[:, :, 1] == 255) &
            (image_with_boxes[:, :, 2] == 0)
        )
        self.assertGreater(len(green_pixels[0]), 0)  # Ci dovrebbero essere pixel verdi

            # Verifica che ci siano pixel bianchi (colore del testo)
        # Usiamo una condizione rilassata per trovare pixel "quasi bianchi"
        white_pixels = np.where(
            (image_with_boxes[:, :, 0] > 250) &
            (image_with_boxes[:, :, 1] > 250) &
            (image_with_boxes[:, :, 2] > 250)
        )
     #   self.assertGreater(len(white_pixels[0]), 0)  # Ci dovrebbero essere pixel quasi bianchi


        # # Debug: Visualizza l'immagine
        # cv2.imshow("Image with Boxes and Labels", image_with_boxes)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    unittest.main()