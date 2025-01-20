import pytest
import numpy as np
import cv2
import os
import tempfile
from saltup.utils.data.image import ColorMode,ImageFormat
from saltup.utils.data.image import load_image

# class TestImageProcessing:
#     @pytest.fixture
#     def sample_image_path(self):
#         """Create a temporary test image"""
#         with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
#             # Create a simple test image (100x100 gradient)
#             img = np.linspace(0, 255, 10000, dtype=np.uint8).reshape(100, 100)
#             cv2.imwrite(tmp.name, img)
#             yield tmp.name
#         os.unlink(tmp.name)
    
#     @pytest.fixture
#     def sample_color_image_path(self):
#         """Create a temporary color test image"""
#         with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
#             # Create a simple color test image
#             img = np.zeros((100, 100, 3), dtype=np.uint8)
#             img[:, :, 0] = 255  # Red channel
#             cv2.imwrite(tmp.name, img)
#             yield tmp.name
#         os.unlink(tmp.name)

#     def test_load_grayscale_image(self, sample_image_path):
        
#         image = load_image(sample_image_path,color_mode=ColorMode.GRAY)
#         assert isinstance(image, np.ndarray)
#         print(image.shape)
#         assert len(image.shape) == 3 and image.shape[-1] == 1
#         assert image.dtype == np.uint8

#     def test_jpg_to_raw_array_grayscale(self, sample_image_path):
#         from saltup.utils.data.image import jpg_to_raw_array
        
#         img_array = jpg_to_raw_array(sample_image_path, grayscale=True)
#         assert isinstance(img_array, np.ndarray)
#         assert len(img_array.shape) == 2
#         assert img_array.dtype == np.uint8

#     def test_jpg_to_raw_array_color(self, sample_color_image_path):
#         from saltup.utils.data.image import jpg_to_raw_array
        
#         img_array = jpg_to_raw_array(sample_color_image_path, grayscale=False)
#         assert isinstance(img_array, np.ndarray)
#         assert len(img_array.shape) == 3
#         assert img_array.shape[2] == 3  # Should have 3 color channels
#         assert img_array.dtype == np.uint8

#     def test_resize_image(self, sample_image_path):
#         from saltup.utils.data.image import load_image, resize_image
#         image = load_image(sample_image_path,color_mode=ColorMode.GRAY)
#         new_size = (50, 50)
#         resized = resize_image(image, new_size)
        
#         assert isinstance(resized, np.ndarray)
#         assert resized.shape[:2] == new_size[::-1]  # OpenCV uses (height, width) order

#     def test_crop_image(self, sample_image_path):
#         from saltup.utils.data.image import  crop_image
        
#         image = load_image(sample_image_path,color_mode=ColorMode.GRAY)

#         crop_window = {
#             'x_min': 20,
#             'y_min': 20,
#             'x_max': 80,
#             'y_max': 80
#         }
        
#         cropped = crop_image(image, crop_window)
#         assert isinstance(cropped, np.ndarray)
#         assert cropped.shape == (60, 60,1)  # 80-20 = 60 for both dimensions

#     def test_save_raw_image(self, sample_image_path, tmp_path):
#         from saltup.utils.data.image import  save_raw_image
        
#         image = load_image(sample_image_path,color_mode=ColorMode.GRAY)

#         output_path = str(tmp_path / "output.raw")
#         save_raw_image(image, output_path)
        
#         assert os.path.exists(output_path)
#         with open(output_path, 'rb') as f:
#             raw_data = f.read()
#         assert len(raw_data) == image.size * image.itemsize

#     def test_save_jpg_image(self, sample_image_path, tmp_path):
#         from saltup.utils.data.image import  save_jpg_image
        
#         image = load_image(sample_image_path,color_mode=ColorMode.GRAY)
#         output_path = str(tmp_path / "output.jpg")
#         save_jpg_image(image, output_path)
        
#         assert os.path.exists(output_path)
#         # Verify we can read it back
#         loaded = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
#         loaded = np.expand_dims(loaded, axis=-1) # we need 3 axis
#         assert loaded is not None
#         assert loaded.shape == image.shape

#     def test_invert_pixel(self, sample_image_path):
#         from saltup.utils.data.image import  invert_pixel
        
#         image = load_image(sample_image_path,color_mode=ColorMode.GRAY)
#         print(image.shape)
#         inverted = invert_pixel(image)
        
#         assert isinstance(inverted, np.ndarray)
#         print(inverted.shape)
#         assert inverted.shape == image.shape
#         # Test that pixels are actually inverted
#         np.testing.assert_array_equal(inverted, 255 - image)


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
        image = image_instance.get_data()
        assert isinstance(image, np.ndarray)
        print(image.shape)
        assert len(image.shape) == 3 and image.shape[-1] == 1  # Grayscale image has shape (H, W, 1)
        assert image.dtype == np.uint8
        assert image_instance.get_color_mode() == ColorMode.GRAY
        assert image_instance.get_image_format() == ImageFormat.HWC

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