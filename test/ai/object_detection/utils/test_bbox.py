import pytest
import numpy as np

from saltup.ai.object_detection.utils.bbox import (
    yolo_to_coco_bbox, coco_to_yolo_bbox, pascalvoc_to_yolo_bbox,
    corners_to_center_format, corners_to_topleft_format,
    center_to_corners_format, center_to_topleft_format,
    topleft_to_center_format, normalize_bbox, absolute_bbox,
    compute_iou, is_normalized, BBoxFormat, NotationFormat
)

class TestBBoxUtils:
    @pytest.fixture
    def image_dimensions(self):
        """Standard image dimensions for testing."""
        return 640, 480  # width, height
        
    @pytest.fixture
    def sample_boxes(self):
        """Sample bounding boxes in different formats."""
        return {
            'yolo': [0.5, 0.5, 0.2, 0.3],         # center_x, center_y, width, height
            'coco': [100, 100, 40, 60],           # x1, y1, width, height
            'pascal': [90, 80, 140, 160],         # x1, y1, x2, y2
            'pascal_norm': [0.2, 0.2, 0.4, 0.5],  # normalized pascal coordinates
            'topleft': [100, 100, 50, 60],        # x1, y1, width, height
            'center': [125, 130, 50, 60]          # center_x, center_y, width, height
        }

    def test_yolo_to_coco_bbox(self, sample_boxes, image_dimensions):
        """Test YOLO to COCO bbox format conversion."""
        img_width, img_height = image_dimensions
        yolo_bbox = sample_boxes['yolo']
        
        coco_bbox = yolo_to_coco_bbox(yolo_bbox, img_width, img_height)
        
        # Basic format checks
        assert len(coco_bbox) == 4
        assert all(isinstance(x, (int, float)) for x in coco_bbox)
        
        # Convert back to verify
        yolo_again = coco_to_yolo_bbox(coco_bbox, img_width, img_height)
        np.testing.assert_array_almost_equal(yolo_bbox, yolo_again)

    def test_coco_to_yolo_bbox(self, sample_boxes, image_dimensions):
        """Test COCO to YOLO bbox format conversion."""
        img_width, img_height = image_dimensions
        coco_bbox = sample_boxes['coco']
        
        yolo_bbox = coco_to_yolo_bbox(coco_bbox, img_width, img_height)
        
        # Verify format
        assert len(yolo_bbox) == 4
        assert all(0 <= x <= 1 for x in yolo_bbox)  # YOLO coordinates should be normalized
        
        # Convert back to verify
        coco_again = yolo_to_coco_bbox(yolo_bbox, img_width, img_height)
        np.testing.assert_array_almost_equal(coco_bbox, coco_again, decimal=0)  # Using decimal=0 due to rounding

    def test_pascalvoc_to_yolo_bbox(self, sample_boxes, image_dimensions):
        """Test Pascal VOC to YOLO bbox format conversion."""
        img_width, img_height = image_dimensions
        pascal_bbox = sample_boxes['pascal']
        
        yolo_bbox = pascalvoc_to_yolo_bbox(pascal_bbox, img_width, img_height)
        
        # Check format and normalization
        assert len(yolo_bbox) == 4
        assert all(0 <= x <= 1 for x in yolo_bbox)
        
        # Verify the center coordinates make sense
        x_center, y_center, w, h = yolo_bbox
        assert 0 < x_center < 1
        assert 0 < y_center < 1
        assert 0 < w < 1
        assert 0 < h < 1

    def test_corners_to_center_format(self, sample_boxes):
        """Test conversion from corners (x1,y1,x2,y2) to center format (xc,yc,w,h)."""
        corners = sample_boxes['pascal']
        
        center = corners_to_center_format(corners)
        
        # Format check
        assert len(center) == 4
        xc, yc, w, h = center
        x1, y1, x2, y2 = corners
        
        # Verify calculations
        assert xc == (x1 + x2) / 2
        assert yc == (y1 + y2) / 2
        assert w == x2 - x1
        assert h == y2 - y1
        
        # Convert back
        corners_again = center_to_corners_format(center)
        np.testing.assert_array_almost_equal(corners, corners_again)

    def test_corners_to_topleft_format(self, sample_boxes):
        """Test conversion from corners to top-left format."""
        corners = sample_boxes['pascal']
        topleft = corners_to_topleft_format(corners)
        
        # Check format
        assert len(topleft) == 4
        x1, y1, w, h = topleft
        x1_orig, y1_orig, x2_orig, y2_orig = corners
        
        # Verify calculations
        assert x1 == x1_orig
        assert y1 == y1_orig
        assert w == x2_orig - x1_orig
        assert h == y2_orig - y1_orig

    def test_center_to_topleft_format(self, sample_boxes):
        """Test conversion from center to top-left format."""
        center = sample_boxes['center']
        topleft = center_to_topleft_format(center)
        
        # Format check
        assert len(topleft) == 4
        x, y, w, h = topleft
        xc, yc, w_orig, h_orig = center
        
        # Verify calculations
        assert x == xc - w_orig/2
        assert y == yc - h_orig/2
        assert w == w_orig
        assert h == h_orig

    def test_topleft_to_center_format(self, sample_boxes):
        """Test conversion from top-left to center format."""
        topleft = sample_boxes['topleft']
        center = topleft_to_center_format(topleft)
        
        # Check format
        assert len(center) == 4
        xc, yc, w, h = center
        x, y, w_orig, h_orig = topleft
        
        # Verify calculations
        assert xc == x + w_orig/2
        assert yc == y + h_orig/2
        assert w == w_orig
        assert h == h_orig

    def test_normalize_bbox(self, sample_boxes, image_dimensions):
        """Test bbox normalization for different formats."""
        img_width, img_height = image_dimensions
        
        # Test corners format
        norm_corners = normalize_bbox(sample_boxes['pascal'], img_width, img_height, format=BBoxFormat.CORNERS)
        assert all(0 <= x <= 1 for x in norm_corners)
        
        # Test top-left format
        norm_topleft = normalize_bbox(sample_boxes['topleft'], img_width, img_height, format=BBoxFormat.TOPLEFT)
        assert all(0 <= x <= 1 for x in norm_topleft)
        
        # Test center format
        norm_center = normalize_bbox(sample_boxes['center'], img_width, img_height, format=BBoxFormat.CENTER)
        assert all(0 <= x <= 1 for x in norm_center)

        # Test invalid format
        with pytest.raises(TypeError):
            normalize_bbox(sample_boxes['pascal'], img_width, img_height, format='invalid')

    def test_absolute_bbox(self, sample_boxes, image_dimensions):
        """Test conversion from normalized to absolute coordinates."""
        img_width, img_height = image_dimensions
        
        # Test with normalized pascal format
        abs_pascal = absolute_bbox(sample_boxes['pascal_norm'], img_width, img_height, format=BBoxFormat.CORNERS)
        assert all(x > 1 for x in abs_pascal)  # Should be in pixels now
        
        # Normalize back to verify
        norm_again = normalize_bbox(abs_pascal, img_width, img_height, format=BBoxFormat.CORNERS)
        np.testing.assert_array_almost_equal(sample_boxes['pascal_norm'], norm_again)

    def test_compute_iou(self):
        """Test IoU calculation between bounding boxes."""
        # Perfect overlap
        box1 = [0, 0, 1, 1]
        np.testing.assert_almost_equal(compute_iou(box1, box1, format=BBoxFormat.CORNERS), 1.0)
        
        # No overlap
        box2 = [2, 2, 3, 3]
        assert compute_iou(box1, box2, format=BBoxFormat.CORNERS) == 0.0
        
        # Partial overlap
        box3 = [0.5, 0.5, 1.5, 1.5]
        iou = compute_iou(box1, box3, format=BBoxFormat.CORNERS)
        assert 0 < iou < 1
        
        # Test with center format
        center_box1 = [0.5, 0.5, 1, 1]  # center_x, center_y, w, h
        center_box2 = [0.5, 0.5, 1, 1]
        np.testing.assert_almost_equal(compute_iou(center_box1, center_box2, format=BBoxFormat.CENTER), 1.0)

    def test_is_normalized(self, sample_boxes):
        """Test bbox normalization check."""
        assert is_normalized(sample_boxes['yolo'])
        assert not is_normalized(sample_boxes['coco'])
        assert not is_normalized(sample_boxes['pascal'])
        assert is_normalized(sample_boxes['pascal_norm'])

if __name__ == '__main__':
    pytest.main(['-v', __file__])