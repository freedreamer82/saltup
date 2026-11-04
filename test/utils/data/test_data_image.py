import pytest
import numpy as np
import cv2
import os
import tempfile
from saltup.utils.data.image import ColorMode,ImageFormat

import pytest
import numpy as np
import cv2
import os
import tempfile
from saltup.utils.data.image import ColorMode, Image  # Import the Image class

class TestImageProcessing:
    @pytest.fixture
    def sample_image_path(self):
        """Create a temporary test image"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            # Create a simple test image (100x100 gradient)
            img = np.linspace(0, 255, 10000, dtype=np.uint8).reshape(100, 100)
            cv2.imwrite(tmp.name, img)
            yield tmp.name
        os.unlink(tmp.name)
    
    @pytest.fixture
    def sample_color_image_path(self):
        """Create a temporary color test image"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            # Create a simple color test image
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img[:, :, 0] = 255  # Red channel
            cv2.imwrite(tmp.name, img)
            yield tmp.name
        os.unlink(tmp.name)

    def test_load_grayscale_image(self, sample_image_path):
        # Use the Image class to load the image
        image_instance = Image(sample_image_path, color_mode=ColorMode.GRAY)
        image = image_instance.get_data(format=ImageFormat.HWC)
        assert isinstance(image, np.ndarray)
        print(image.shape)
        assert len(image.shape) == 3 and image.shape[-1] == 1  # Grayscale image has shape (H, W, 1)
        assert image.dtype == np.uint8
        assert image_instance.get_color_mode() == ColorMode.GRAY

    def test_jpg_to_raw_array_grayscale(self, sample_image_path):
        # Use the class method to convert the image to a raw array
        img_array = Image.jpg_to_raw(sample_image_path, grayscale=True)
        assert isinstance(img_array, np.ndarray)
        assert len(img_array.shape) == 2  # Grayscale image has shape (H, W)
        assert img_array.dtype == np.uint8

    def test_jpg_to_raw_array_color(self, sample_color_image_path):
        # Use the class method to convert the image to a raw array
        img_array = Image.jpg_to_raw(sample_color_image_path, grayscale=False)
        assert isinstance(img_array, np.ndarray)
        assert len(img_array.shape) == 3  # Color image has shape (H, W, 3)
        assert img_array.shape[2] == 3  # Should have 3 color channels
        assert img_array.dtype == np.uint8

    def test_resize_image(self, sample_image_path):
        # Load the image using the Image class
        image_instance = Image(sample_image_path, color_mode=ColorMode.GRAY)
        image = image_instance.get_data()
        new_size = (50, 50)
        # Use the class method to resize the image
        resized = Image.resize_image(image, new_size)
        
        assert isinstance(resized, np.ndarray)
        assert resized.shape[:2] == new_size[::-1]  # OpenCV uses (height, width) order

    def test_crop_image(self, sample_image_path):
        # Load the image using the Image class
        image_instance = Image(sample_image_path, color_mode=ColorMode.GRAY)
        image = image_instance.get_data()

        crop_window = {
            'x_min': 20,
            'y_min': 20,
            'x_max': 80,
            'y_max': 80
        }
        
        # Use the class method to crop the image
        cropped = Image.crop_image(image, crop_window)
        assert isinstance(cropped, np.ndarray)
        assert cropped.shape == (60, 60, 1)  # 80-20 = 60 for both dimensions, and 1 channel for grayscale

    def test_save_raw_image(self, sample_image_path, tmp_path):
        # Load the image using the Image class
        image_instance = Image(sample_image_path, color_mode=ColorMode.GRAY)
        image = image_instance.get_data()

        output_path = str(tmp_path / "output.raw")
        # Use the class method to save the image in raw format
        Image.save_raw_image(image, output_path)
        
        assert os.path.exists(output_path)
        with open(output_path, 'rb') as f:
            raw_data = f.read()
        assert len(raw_data) == image.size * image.itemsize

    def test_save_jpg_image(self, sample_image_path, tmp_path):
        # Load the image using the Image class
        image_instance = Image(sample_image_path, color_mode=ColorMode.GRAY)
        image = image_instance.get_data()
        output_path = str(tmp_path / "output.jpg")
        # Use the class method to save the image as JPG
        Image.save_jpg_image(image, output_path)
        
        assert os.path.exists(output_path)
        # Verify we can read it back
        loaded = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
        loaded = np.expand_dims(loaded, axis=-1)  # Ensure it has 3 axes
        assert loaded is not None
        assert loaded.shape == image.shape

    def test_invert_pixel(self, sample_image_path):
        # Load the image using the Image class
        image_instance = Image(sample_image_path, color_mode=ColorMode.GRAY)
        image = image_instance.get_data()
        print(image.shape)
        # Use the class method to invert the pixel values
        inverted = Image.invert_pixel(image)
        
        assert isinstance(inverted, np.ndarray)
        print(inverted.shape)
        assert inverted.shape == image.shape
        # Test that pixels are actually inverted
        np.testing.assert_array_equal(inverted, 255 - image)
    
    # Test cases for pad_image class method
    def test_pad_image_smaller_than_target(self):
        # Input image (2x2) in HWC format
        image = np.array([
            [[1], [2]],
            [[3], [4]]
        ], dtype=np.float32)  # HWC format
        target_h, target_w = 4, 4

        # Expected output: padded image with value 114 (in HWC format)
        expected_output = np.array([
            [[1], [2], [114], [114]],
            [[3], [4], [114], [114]],
            [[114], [114], [114], [114]],
            [[114], [114], [114], [114]]
        ], dtype=np.float32)

        # Test
        result = Image.pad_image(image, target_h, target_w, ImageFormat.HWC)
        np.testing.assert_array_equal(result, expected_output)

    def test_pad_image_larger_than_target(self):
        # Input image (4x4) larger than target size (2x2)
        image = np.array([[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]], dtype=np.float32)  # CHW format
        target_h, target_w = 2, 2

        # Expected output: original image (no padding needed)
        expected_output = np.array([[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]], dtype=np.float32)

        # Test
        result = Image.pad_image(image, target_h, target_w, ImageFormat.CHW)
        np.testing.assert_array_equal(result, expected_output)

    def test_pad_image_exact_target_size(self):
        # Input image (3x3) matches target size (3x3)
        image = np.array([[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]], dtype=np.float32)  # CHW format
        target_h, target_w = 3, 3

        # Expected output: original image (no padding needed)
        expected_output = image

        # Test
        result = Image.pad_image(image, target_h, target_w, ImageFormat.CHW)
        np.testing.assert_array_equal(result, expected_output)

    def test_pad_image_hwc_format(self):
        # Input image (2x2) in HWC format
        image = np.array([
            [[1], [2]],
            [[3], [4]]
        ], dtype=np.float32)  # HWC format
        target_h, target_w = 4, 4

        # Expected output: padded image with value 114 (in HWC format)
        expected_output = np.array([
            [[1], [2], [114], [114]],
            [[3], [4], [114], [114]],
            [[114], [114], [114], [114]],
            [[114], [114], [114], [114]]
        ], dtype=np.float32)

        # Test
        result = Image.pad_image(image, target_h, target_w, ImageFormat.HWC)
        np.testing.assert_array_equal(result, expected_output)

    def test_pad_image_multichannel(self):
        # Input image (2x2x3) in HWC format (3 channels)
        image = np.array([
            [[1, 10, 100], [2, 20, 200]],
            [[3, 30, 300], [4, 40, 400]]
        ], dtype=np.float32)  # HWC format
        target_h, target_w = 4, 4

        # Expected output: padded image with value 114 (in HWC format)
        expected_output = np.array([
            [[1, 10, 100], [2, 20, 200], [114, 114, 114], [114, 114, 114]],
            [[3, 30, 300], [4, 40, 400], [114, 114, 114], [114, 114, 114]],
            [[114, 114, 114], [114, 114, 114], [114, 114, 114], [114, 114, 114]],
            [[114, 114, 114], [114, 114, 114], [114, 114, 114], [114, 114, 114]]
        ], dtype=np.float32)

        # Test
        result = Image.pad_image(image, target_h, target_w, ImageFormat.HWC)
        np.testing.assert_array_equal(result, expected_output)

class TestImageEncoding:
    @pytest.fixture
    def color_image(self):
        """RGB Image instance with distinct per-channel values"""
        img = np.zeros((20, 30, 3), dtype=np.uint8)
        img[:, :, 0] = 200  # R
        img[:, :, 1] = 100  # G
        img[:, :, 2] = 50   # B
        return Image(img, color_mode=ColorMode.RGB)

    @pytest.fixture
    def gray_image(self):
        img = np.linspace(0, 255, 20 * 30, dtype=np.uint8).reshape(20, 30, 1)
        return Image(img, color_mode=ColorMode.GRAY)

    @pytest.mark.parametrize(
        "method, kwargs, magic",
        [
            ("to_jpeg", {}, b"\xff\xd8\xff"),
            ("to_png", {}, b"\x89PNG"),
            ("to_bmp", {}, b"BM"),
            ("to_webp", {}, b"RIFF"),
            ("to_tiff", {}, b"II*\x00"),
        ],
    )
    def test_encode_returns_buffer_with_expected_signature(self, color_image, method, kwargs, magic):
        buffer = getattr(color_image, method)(**kwargs)
        assert isinstance(buffer, np.ndarray)
        assert buffer.ndim == 1 and buffer.dtype == np.uint8
        assert buffer.tobytes().startswith(magic)

    @pytest.mark.parametrize("method, ext", [
        ("to_png", "png"), ("to_bmp", "bmp"), ("to_tiff", "tif"),
    ])
    def test_lossless_roundtrip_preserves_rgb_pixels(self, color_image, method, ext, tmp_path):
        out = tmp_path / f"img.{ext}"
        buffer = getattr(color_image, method)(save_path=out)

        assert out.exists()
        assert out.read_bytes() == buffer.tobytes()

        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)  # BGR
        decoded_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(decoded_rgb, color_image.get_data())

    def test_png_roundtrip_grayscale(self, gray_image):
        decoded = cv2.imdecode(gray_image.to_png(), cv2.IMREAD_UNCHANGED)
        assert decoded.ndim == 2
        np.testing.assert_array_equal(decoded, gray_image.get_data()[:, :, 0])

    def test_jpeg_quality_affects_size(self, gray_image):
        assert gray_image.to_jpeg(quality=10).size < gray_image.to_jpeg(quality=100).size

    def test_png_compression_affects_size(self, gray_image):
        assert gray_image.to_png(compression=9).size <= gray_image.to_png(compression=0).size

    def test_encode_accepts_string_format(self, color_image):
        assert color_image._encode("png").tobytes().startswith(b"\x89PNG")
        assert color_image._encode(".BMP").tobytes().startswith(b"BM")

    def test_encode_rejects_unsupported_format(self, color_image):
        with pytest.raises(ValueError):
            color_image._encode("mp4")
        with pytest.raises(ValueError):
            color_image._encode("notaformat")

    def test_invalid_parameters(self, color_image):
        with pytest.raises(ValueError):
            color_image.to_jpeg(quality=101)
        with pytest.raises(ValueError):
            color_image.to_png(compression=10)

    def test_encode_does_not_mutate_image(self, color_image):
        before = color_image.get_data().copy()
        color_image.to_png()
        color_image.to_bmp()
        np.testing.assert_array_equal(color_image.get_data(), before)
        assert color_image.get_color_mode() == ColorMode.RGB
