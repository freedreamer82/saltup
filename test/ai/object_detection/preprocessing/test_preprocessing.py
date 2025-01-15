import pytest
import numpy as np
import os

from saltup.ai.object_detection.preprocessing import (
    PreprocessingFactory,
    PreprocessingType,
    Preprocessing
)
from saltup.ai.object_detection.preprocessing.impl import (
    AnchorsBasedPreprocess,
    DamoPreprocessing,
    SupergradPreprocess,
    SupergradPreprocessType,
    UltralyticsPreprocess
)


class TestBasePreprocessing:
    """Test the abstract base preprocessing class."""

    class ConcretePreprocessing(Preprocessing):
        """Concrete class for testing abstract base class."""

        def __call__(self):
            return super().__call__()

    def test_validate_input_none(self):
        """Test validation of None input."""
        processor = self.ConcretePreprocessing()
        with pytest.raises(ValueError, match="Input image cannot be None"):
            processor._validate_input(None)

    def test_validate_input_wrong_type(self):
        """Test validation of non-numpy array input."""
        processor = self.ConcretePreprocessing()
        with pytest.raises(TypeError, match="Input must be numpy array"):
            processor._validate_input([1, 2, 3])

    def test_abstract_process_method(self):
        """Test that abstract process method raises NotImplementedError."""
        processor = self.ConcretePreprocessing()
        with pytest.raises(NotImplementedError):
            processor.__call__()


class TestPreprocessingFactory:
    """Test the preprocessing factory class."""

    def test_valid_processor_creation(self):
        """Test creation of all valid processor types."""
        valid_types = [
            (PreprocessingType.ANCHORS_BASED, AnchorsBasedPreprocess),
            (PreprocessingType.ULTRALITICS, UltralyticsPreprocess),
            (PreprocessingType.SUPERGRAD, SupergradPreprocess),
            (PreprocessingType.DAMO, DamoPreprocessing)
        ]

        for proc_type, expected_class in valid_types:
            processor = PreprocessingFactory.create(proc_type)
            assert isinstance(processor, expected_class)

    def test_invalid_processor_type(self):
        """Test factory behavior with invalid processor type."""
        with pytest.raises(ValueError, match="Unknown processor type"):
            PreprocessingFactory.create(999)


class TestAnchorsBasedPreprocess:
    """Test the anchors-based preprocessing implementation."""

    @pytest.fixture
    def processor(self):
        return AnchorsBasedPreprocess()

    @pytest.fixture
    def sample_image(self):
        """Create a sample grayscale image."""
        return np.zeros((100, 150), dtype=np.uint8)

    def test_process_valid_image(self, processor, sample_image):
        """Test processing of valid grayscale image."""
        target_shape = (224, 224)
        result = processor(sample_image, target_shape)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, target_shape[0], target_shape[1], 1)
        assert result.dtype == np.float32
        assert 0 <= result.min() <= result.max() <= 1

    def test_custom_normalization(self, processor, sample_image):
        """Test processing with custom normalization method."""
        def custom_norm(img):
            return (img - 128.0) / 128.0

        target_shape = (224, 224)
        processor.normalize_method = custom_norm
        result = processor(sample_image, target_shape)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert -1 <= result.min() <= result.max() <= 1


class TestDamoPreprocessing:
    """Test the DAMO preprocessing implementation."""

    @pytest.fixture
    def processor(self):
        return DamoPreprocessing()

    @pytest.fixture
    def sample_image(self):
        """Create a sample BGR image."""
        return np.zeros((100, 150, 3), dtype=np.uint8)

    def test_process_valid_image(self, processor, sample_image):
        """Test processing of valid BGR image."""
        target_shape = (224, 224)
        result = processor(sample_image, target_shape)

        assert isinstance(result, np.ndarray)
        assert result.shape == (
            1, 3, target_shape[0], target_shape[1])  # NCHW format
        assert result.dtype == np.float32

    def test_pad_image(self, processor):
        """Test image padding functionality."""
        test_image = np.zeros((3, 100, 150))  # CHW format
        target_size = (200, 250)

        padded = processor.pad_image(test_image, target_size)

        assert padded.shape == (3, 200, 250)
        # Original content preserved
        assert np.all(padded[:, :100, :150] == test_image)
        assert np.all(padded[:, 100:, 150:] == 0)  # Padding is zeros


class TestSupergradPreprocess:
    """Test the Supergrad preprocessing implementation."""

    @pytest.fixture
    def processor(self):
        return SupergradPreprocess()

    @pytest.fixture
    def sample_image(self):
        """Create a sample BGR image."""
        return np.zeros((100, 150, 3), dtype=np.uint8)

    def test_process_base_pipeline(self, processor, sample_image):
        """Test standard preprocessing pipeline."""
        target_shape = (224, 224)
        result = processor(sample_image, target_shape)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3, target_shape[0], target_shape[1])
        assert result.dtype == np.float32
        assert 0 <= result.min() <= result.max() <= 1

    def test_process_qat_pipeline(self, processor, sample_image):
        """Test QAT preprocessing pipeline."""
        target_shape = (224, 224)
        processor.preprocess_type = SupergradPreprocessType.QAT
        result = processor(
            sample_image,
            target_shape,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (target_shape[0], target_shape[1], 3)
        assert result.dtype == np.uint8

    def test_resize_image_positions(self, processor, sample_image):
        """Test image positioning during resize."""
        target_shape = (300, 400)

        # Test TOP_LEFT positioning
        result_top_left = processor.resize_image_and_black_container_rgb(
            sample_image,
            final_width=target_shape[1],
            final_height=target_shape[0],
            img_position=processor.ImagePosition.TOP_LEFT
        )

        # Test CENTER positioning
        result_center = processor.resize_image_and_black_container_rgb(
            sample_image,
            final_width=target_shape[1],
            final_height=target_shape[0],
            img_position=processor.ImagePosition.CENTER
        )

        assert result_top_left.shape == (*target_shape, 3)
        assert result_center.shape == (*target_shape, 3)
        assert not np.array_equal(result_top_left, result_center)


class TestUltraliticsPreprocess:
    """Test the Ultralitics preprocessing implementation."""

    @pytest.fixture
    def processor(self):
        return UltralyticsPreprocess()

    @pytest.fixture
    def sample_image(self):
        """Create a sample BGR image."""
        return np.zeros((100, 150, 3), dtype=np.uint8)
    
    @pytest.fixture
    def reference_result(self, root_dir):
        """Load reference test data."""
        # Specify correct path relative to test file
        path = os.path.join(str(root_dir), 'results/result_process_with_overrides_2.npz')
        data = np.load(path)
        return data['arr_0']  # Access specific array from npz

    def test_process_default_settings(self, processor, sample_image):
        """Test processing with default settings."""
        result = processor(sample_image)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3, 640, 640)  # Default shape
        assert result.dtype == np.float32
        assert 0 <= result.min() <= result.max() <= 1

    def test_process_with_overrides_1(self, processor, sample_image):
        """Verify preprocessing with custom parameters.
        
        Tests minimum rectangle and scale fill options.
        """
        result = processor(
            sample_image,
            target_shape=(416, 416),
            auto=True,  # Minimum rectangle
            scale_fill=True
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3, 288, 416)
        assert result.dtype == np.float32
    
    def test_process_with_overrides_2(self, processor, sample_image, reference_result):
        """Verify preprocessing matches reference output.
        
        Tests scale down and centering options.
        """
        result = processor(
            sample_image,
            target_shape=(416, 416),
            scale_up=False,
            center=True
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3, 416, 416)
        np.testing.assert_array_equal(result, reference_result)
        assert result.dtype == np.float32
    
    def test_letterbox_scaling(self, processor, sample_image):
        """Test letterbox scaling behavior."""
        result = processor.letterbox(sample_image, (300, 300))

        assert isinstance(result, np.ndarray)
        # Height and width should match target
        assert result.shape[:2] == (300, 300)
        assert result.shape[2] == 3  # Should preserve channels
