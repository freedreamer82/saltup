import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from saltup.ai.object_detection.utils.anchor_based_model import (
   compute_anchors,
   convert_to_grid_format,
   plot_anchors,
   compute_anchor_iou
)

class TestAnchorIoU:
    @pytest.fixture
    def sample_anchors(self):
        """Sample anchor dimensions for testing."""
        return [
            np.array([0.2, 0.3]),  # Small anchor
            np.array([0.5, 0.5]),  # Medium square anchor
            np.array([0.8, 0.6])   # Large anchor
        ]
    
    def test_identical_anchors(self, sample_anchors):
        """Test IoU between identical anchors should be 1."""
        for anchor in sample_anchors:
            iou = compute_anchor_iou(anchor, anchor)
            assert np.isclose(iou, 1.0)
    
    def test_non_overlapping_anchors(self):
        """Test IoU between completely different anchors should be 0."""
        anchor1 = np.array([0.1, 0.1])  # Very small
        anchor2 = np.array([1.0, 1.0])  # Very large
        iou = compute_anchor_iou(anchor1, anchor2)
        assert iou < 0.1  # Should be very small but not exactly 0 due to eps
        
    def test_partial_overlap(self):
        """Test IoU with partial overlap."""
        anchor1 = np.array([0.3, 0.3])
        anchor2 = np.array([0.2, 0.2])
        iou = compute_anchor_iou(anchor1, anchor2)
        # IoU should be: (0.2 * 0.2) / (0.3 * 0.3 + 0.2 * 0.2 - 0.2 * 0.2)
        expected_iou = 0.2 * 0.2 / (0.3 * 0.3 + 0.2 * 0.2 - 0.2 * 0.2)
        assert np.isclose(iou, expected_iou)
    
    def test_input_validation(self):
        """Test with various input types."""
        # Test with lists
        iou1 = compute_anchor_iou([0.5, 0.5], [0.5, 0.5])
        assert np.isclose(iou1, 1.0)
        
        # Test with numpy arrays
        iou2 = compute_anchor_iou(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
        assert np.isclose(iou2, 1.0)
        
        # Test with mixed types
        iou3 = compute_anchor_iou([0.5, 0.5], np.array([0.5, 0.5]))
        assert np.isclose(iou3, 1.0)
        
    @pytest.mark.parametrize("anchor1,anchor2,expected", [
        ((0.5, 0.5), (0.5, 0.5), 1.0),           # Same size
        ((0.2, 0.2), (0.4, 0.4), 0.25),          # One contains other
        ((0.3, 0.3), (0.3, 0.6), 0.5),           # Same width, different height
        ((0.1, 0.1), (0.2, 0.2), 0.25),          # Similar proportions
    ])
    def test_specific_cases(self, anchor1, anchor2, expected):
        """Test specific IoU cases with known results."""
        iou = compute_anchor_iou(np.array(anchor1), np.array(anchor2))
        assert np.isclose(iou, expected, rtol=1e-2)

class TestAnchorBasedModel:
    @pytest.fixture
    def sample_boxes(self):
        """Sample boxes for testing in (width, height) format."""
        return np.array([
            [0.2, 0.3],
            [0.4, 0.5], 
            [0.3, 0.2],
            [0.5, 0.4],
            [0.1, 0.1]
        ])

    @pytest.fixture
    def sample_annotations(self):
        """Sample annotations in YOLO format (x_center, y_center, width, height)."""
        return np.array([
            [0.5, 0.5, 0.2, 0.3],  # Centered box
            [0.2, 0.3, 0.1, 0.1],  # Top-left box
            [0.8, 0.7, 0.3, 0.2]   # Bottom-right box
        ])

    def test_compute_anchors_basic(self, sample_boxes):
        """Test basic anchor computation."""
        num_anchors = 3
        anchors = compute_anchors(sample_boxes, num_anchors)
        
        assert isinstance(anchors, np.ndarray)
        assert anchors.shape == (num_anchors, 2)
        assert np.all((anchors >= 0) & (anchors <= 1))

    def test_compute_anchors_validation(self, sample_boxes):
        """Test input validation for compute_anchors."""
        # Test invalid input type
        with pytest.raises(TypeError):
            compute_anchors([[0.1, 0.1]], 2)  # Not numpy array
            
        # Test invalid shapes
        with pytest.raises(ValueError):
            compute_anchors(np.array([0.1, 0.1]), 2)  # 1D array
            
        # Test empty array
        with pytest.raises(ValueError):
            compute_anchors(np.array([]).reshape(0, 2), 2)
            
        # Test num_anchors validation
        with pytest.raises(ValueError):
            compute_anchors(sample_boxes, 0)  # Too few anchors
            
        with pytest.raises(ValueError):
            compute_anchors(sample_boxes, len(sample_boxes) + 1)  # Too many anchors

    def test_convert_to_grid_format(self, sample_annotations):
        """Test conversion to grid format."""
        class_labels = [0, 1, 2]  # One label per box
        grid_size = (13, 13)
        anchors = [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)]
        num_classes = 3
        
        grid_labels = convert_to_grid_format(
            sample_annotations, 
            class_labels,
            grid_size,
            anchors, 
            num_classes
        )
        
        # Check output shape
        expected_shape = (1, 13, 13, len(anchors), 5 + num_classes)
        assert grid_labels.shape == expected_shape
        
        # Check value ranges
        assert np.all(grid_labels[..., 4] >= 0)  # Objectness scores
        assert np.all(grid_labels[..., 4] <= 1)
        
        # Check one-hot encoding
        class_probs = grid_labels[..., 5:]
        assert np.all((class_probs == 0) | (class_probs == 1))
        assert np.sum(class_probs) == len(sample_annotations)  # One class per box

    @pytest.mark.mpl_image_compare
    def test_plot_anchors(self, sample_boxes, monkeypatch):
        """Test anchor visualization."""
        # Mock plt.show to prevent display
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        # Use first three boxes as anchors
        anchors = sample_boxes[:3]
        image_size = (416, 416)
        
        # Create figure without displaying
        plt.ioff()  # Turn off interactive mode
        plot_anchors(anchors, image_size, "Test Anchors")
        fig = plt.gcf()
        
        # Basic validation of plot components
        assert len(fig.axes) == 1  # One subplot
        ax = fig.axes[0]
        
        # Check axis limits
        assert ax.get_xlim() == (0, 416)
        assert ax.get_ylim() == (416, 0)  # Inverted y-axis
        
        # Check number of rectangles (anchors)
        rectangles = [p for p in ax.patches if isinstance(p, patches.Rectangle)]
        assert len(rectangles) == len(anchors)
        
        plt.close(fig)

    def test_convert_to_grid_format_edge_cases(self):
        """Test grid conversion with edge cases."""
        # Empty input
        empty_result = convert_to_grid_format(
            boxes=np.array([]),
            class_labels=[],
            grid_size=(7, 7),
            anchors=[(0.1, 0.1)],
            num_classes=1
        )
        assert empty_result.shape == (1, 7, 7, 1, 6)
        assert np.all(empty_result == 0)
        
        # Single box at grid boundary
        boundary_box = np.array([[1.0, 1.0, 0.1, 0.1]])  # Right-bottom corner
        boundary_result = convert_to_grid_format(
            boxes=boundary_box,
            class_labels=[0],
            grid_size=(7, 7),
            anchors=[(0.1, 0.1)],
            num_classes=1
        )
        assert boundary_result.shape == (1, 7, 7, 1, 6)