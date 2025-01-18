import unittest
from typing import List, Tuple
from saltup.ai.object_detection.utils.bbox import BBox, BBoxFormat, IoUType
import json
import xml.etree.ElementTree as ET

class TestBBox(unittest.TestCase):

    def setUp(self):
        # Setup common test data
        self.corners_coords = [10, 20, 50, 60]  # x1, y1, x2, y2
        self.center_coords = [30, 40, 40, 40]   # x_center, y_center, width, height
        self.topleft_coords = [10, 20, 40, 40]  # x_min, y_min, width, height
        self.img_width = 100
        self.img_height = 100

    def test_initialization(self):
        bbox = BBox(self.corners_coords, BBoxFormat.CORNERS, self.img_width, self.img_height)
        self.assertEqual(bbox.coordinates, self.corners_coords)
        self.assertEqual(bbox.format, BBoxFormat.CORNERS)
        self.assertEqual(bbox.img_width, self.img_width)
        self.assertEqual(bbox.img_height, self.img_height)

    def test_from_yolo_file(self):
        # Create a mock YOLO file
        yolo_file = "mock_yolo.txt"
        with open(yolo_file, 'w') as f:
            f.write("0 0.3 0.4 0.2 0.2\n")

        bboxes = BBox.from_yolo_file(yolo_file, self.img_width, self.img_height)
        # Verifica che il risultato sia una tupla
        self.assertIsInstance(bboxes, tuple)
        
        # Verifica che la tupla contenga due elementi
        self.assertEqual(len(bboxes), 2)
        
        # Verifica che il primo elemento sia una lista di BBox
        self.assertIsInstance(bboxes[0], list)
        if bboxes[0]:  # Se la lista non è vuota
            self.assertIsInstance(bboxes[0][0], BBox)
            
        self.assertEqual(bboxes[0][0].get_coordinates(), [0.3, 0.4, 0.2, 0.2])
        self.assertEqual(bboxes[0][0].get_format(), BBoxFormat.CENTER)

    def test_from_coco_file(self):
        # Create a mock COCO file
        coco_file = "mock_coco.json"
        coco_data = {
            "annotations": [
                {"image_id": 1, "bbox": [10, 20, 40, 40]},
                {"image_id": 2, "bbox": [30, 40, 20, 20]}
            ]
        }
        with open(coco_file, 'w') as f:
            json.dump(coco_data, f)

        bboxes = BBox.from_coco_file(coco_file, 1)
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(bboxes[0].coordinates, [10, 20, 40, 40])
        self.assertEqual(bboxes[0].format, BBoxFormat.TOPLEFT)

    def test_from_pascal_voc_file(self):
        # Create a mock Pascal VOC file
        pascal_voc_file = "mock_pascal_voc.xml"
        root = ET.Element("annotation")
        obj = ET.SubElement(root, "object")
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = "10"
        ET.SubElement(bndbox, "ymin").text = "20"
        ET.SubElement(bndbox, "xmax").text = "50"
        ET.SubElement(bndbox, "ymax").text = "60"
        tree = ET.ElementTree(root)
        tree.write(pascal_voc_file)

        bboxes = BBox.from_pascal_voc_file(pascal_voc_file)
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(bboxes[0].coordinates, [10, 20, 50, 60])
        self.assertEqual(bboxes[0].format, BBoxFormat.CORNERS)

    def test_get_coordinates(self):
        bbox = BBox(self.corners_coords, BBoxFormat.CORNERS, self.img_width, self.img_height)
        self.assertEqual(bbox.get_coordinates(BBoxFormat.CENTER), (30.0, 40.0, 40, 40))  # Usa una tupla

    def test_set_coordinates(self):
        bbox = BBox(self.corners_coords, BBoxFormat.CORNERS, self.img_width, self.img_height)
        bbox.set_coordinates([30, 40, 40, 40], BBoxFormat.CENTER)
        self.assertEqual(bbox.coordinates, [30, 40, 40, 40])
        self.assertEqual(bbox.format, BBoxFormat.CENTER)

    def test_normalize(self):
        bbox = BBox(self.corners_coords, BBoxFormat.CORNERS, self.img_width, self.img_height)
        self.assertEqual(bbox.to_yolo(), (0.3, 0.4, 0.4, 0.4)) 

    def test_absolute(self):
        bbox = BBox([0.1, 0.2, 0.5, 0.6], BBoxFormat.CORNERS, self.img_width, self.img_height)
        self.assertEqual(bbox.absolute(), (10, 20, 50, 60))  

    def test_to_yolo(self):
        bbox = BBox(self.corners_coords, BBoxFormat.CORNERS, self.img_width, self.img_height)
        self.assertEqual(bbox.to_yolo(), (0.3, 0.4, 0.4, 0.4))  

    def test_to_coco(self):
        bbox = BBox(self.corners_coords, BBoxFormat.CORNERS, self.img_width, self.img_height)
        self.assertEqual(bbox.to_coco(), (10, 20, 40, 40))  

    def test_to_pascal_voc(self):
        bbox = BBox(self.corners_coords, BBoxFormat.CORNERS, self.img_width, self.img_height)
        self.assertEqual(bbox.to_pascal_voc(), (10, 20, 50, 60)) 

    def test_compute_iou(self):
        bbox1 = BBox([10, 20, 50, 60], BBoxFormat.CORNERS, self.img_width, self.img_height)
        bbox2 = BBox([30, 40, 70, 80], BBoxFormat.CORNERS, self.img_width, self.img_height)
        iou = bbox1.compute_iou(bbox2)
        self.assertAlmostEqual(iou, 0.142857, places=5)

    def tearDown(self):
        # Clean up any mock files created
        import os
        if os.path.exists("mock_yolo.txt"):
            os.remove("mock_yolo.txt")
        if os.path.exists("mock_coco.json"):
            os.remove("mock_coco.json")
        if os.path.exists("mock_pascal_voc.xml"):
            os.remove("mock_pascal_voc.xml")

if __name__ == '__main__':
    unittest.main()