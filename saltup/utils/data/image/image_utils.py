import numpy as np
import cv2
from enum import IntEnum, auto ,Enum
from pathlib import Path
from typing import Union
import random 
from typing import Union, Optional
from pathlib import Path
import numpy as np
import copy
import cv2


class ColorsBGR(Enum):
    RED = (0, 0, 255)        # Rosso in formato BGR
    GREEN = (0, 255, 0)      # Verde in formato BGR
    BLUE = (255, 0, 0)       # Blu in formato BGR
    CYAN = (255, 255, 0)     # Ciano in formato BGR
    MAGENTA = (255, 0, 255)  # Magenta in formato BGR
    YELLOW = (0, 255, 255)   # Giallo in formato BGR
    ORANGE = (0, 165, 255)   # Arancione in formato BGR
    PURPLE = (128, 0, 128)   # Viola in formato BGR
    WHITE = (255, 255, 255)  # Bianco in formato BGR
    BLACK = (0, 0, 0)        # Nero in formato BGR

    def to_rgb(self):
        """
        Convert the BGR color to RGB format.
        
        Returns:
            tuple: The color in RGB format.
        """
        return self.value[::-1]  # Reverse the BGR tuple to get RGB

class ColorMode(IntEnum):
    RGB = auto()
    BGR = auto()
    GRAY = auto()

class ImageFormat(IntEnum):
    HWC = auto()  # Height, Width, Channels (default)
    CHW = auto()  # Channels, Height, Width


def generate_random_bgr_colors(num_colors):
    """
    Generates a list of distinct colors in BGR format. If the number of requested colors
    exceeds the number of predefined colors in the ColorsBGR enum, the colors are reused
    in a cyclic manner.
    
    Args:
        num_colors (int): Number of colors to generate.
    
    Returns:
        list: A list of colors in BGR format.
    """
    # Extract predefined colors from the ColorsBGR enum
    predefined_colors = [color.value for color in ColorsBGR]
    
    # If the number of requested colors is less than or equal to the predefined colors,
    # return a subset of the predefined colors.
    if num_colors <= len(predefined_colors):
        return predefined_colors[:num_colors]
    
    # If more colors are needed, reuse the predefined colors in a cyclic manner.
    colors = []
    for i in range(num_colors):
        color = predefined_colors[i % len(predefined_colors)]  # Cycle through the predefined colors
        colors.append(color)
    
    return colors


class Image:
    def __init__(
        self,
        image_input: Union[str, Path, np.ndarray],  # Accepts either a file path or a NumPy array
        color_mode: ColorMode = ColorMode.BGR
        ):
    
        """
        Initialize an Image instance.

        Args:
            image_input: Path to the image file (str or Path) or a NumPy array containing the image data.
            color_mode: Color mode of the image (BGR, RGB, or GRAY). Default is BGR.
            image_format: Format of the image (HWC or CHW). Default is HWC.
        """
        self.color_mode = color_mode

        # Check if the input is a NumPy array
        if isinstance(image_input, np.ndarray):
             self.image = self._process_image_data(image_input)
        else:
            # Otherwise, treat it as a file path
            self.image_path = Path(image_input) if isinstance(image_input, str) else image_input
            self.image = self._load_image()


    def _process_image_data(self, image_data: np.ndarray)  -> np.ndarray :
        """
        Process the provided NumPy array to ensure it matches the desired color mode and format.

        Args:
            image_data: NumPy array containing the image data.

        Returns:
            Processed image as a NumPy array with shape (h, w, 1) for grayscale or (h, w, 3) for RGB/BGR.
        """
        if not isinstance(image_data, np.ndarray):
            raise ValueError("image_data must be a NumPy array.")

        self.image = self.convert_image_format(image_data, ImageFormat.HWC)  # Ensure the image is in HWC format because opencv works with HWC on that

        # Convert the image to the desired color mode
        try:
            if self.color_mode == ColorMode.RGB:
                if len(self.image .shape) == 3 and self.image .shape[-1] == 3:  # If already RGB, do nothing
                    pass
                else:
                    # Convert BGR to RGB if necessary
                    self.image  = cv2.cvtColor(self.image , cv2.COLOR_BGR2RGB)
            elif self.color_mode == ColorMode.GRAY:
                if len(self.image .shape) == 2:  # If grayscale with shape (h, w), expand to (h, w, 1)
                    self.image = np.expand_dims(self.image , axis=-1)
                elif len(self.image .shape) == 3 and self.image .shape[-1] == 1:  # If already (h, w, 1), do nothing
                    pass
                else:
                    # If it's a 3-channel image but not grayscale, convert to grayscale
                    self.image  = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                    self.image  = np.expand_dims(self.image, axis=-1)
        except cv2.error as e:
            raise ValueError(f"Error converting image color mode: {e}")
            
        return self.image
  

    def _load_image(self) -> np.ndarray:
        """
        Load an image from the specified path and process it.

        Returns:
            Loaded and processed image as a NumPy array with shape (h, w, 1) for grayscale or (h, w, 3) for RGB/BGR.
        """
        # Check if the image file exists
        if not self.image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {self.image_path}")

        # Load the image using OpenCV
        image = cv2.imread(str(self.image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {self.image_path}")

        # Process the loaded image to ensure it matches the desired color mode and format
        return self._process_image_data(image)


    def convert_color_mode(self, new_color_mode: ColorMode):
        """
        Convert the image to a new color mode in-place, ensuring the output has the correct shape
        based on the image format (HWC or CHW).

        Args:
            new_color_mode: The target color mode (BGR, RGB, or GRAY).
        """
        if self.color_mode == new_color_mode:
            return  # No conversion needed

        # Convert the image to the new color mode
        try:
                # Convert in HWC format
                if new_color_mode == ColorMode.RGB:
                    if self.color_mode == ColorMode.BGR:
                        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                    elif self.color_mode == ColorMode.GRAY:
                        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
                elif new_color_mode == ColorMode.BGR:
                    if self.color_mode == ColorMode.RGB:
                        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
                    elif self.color_mode == ColorMode.GRAY:
                        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
                elif new_color_mode == ColorMode.GRAY:
                    if self.color_mode == ColorMode.RGB:
                        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
                    elif self.color_mode == ColorMode.BGR:
                        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                    # Ensure grayscale image has shape (h, w, 1) in HWC format
                    self.image = np.expand_dims(self.image, axis=-1)

        except cv2.error as e:
            raise ValueError(f"Error converting image color mode: {e}")

        # Ensure the image has the correct number of dimensions
        if len(self.image.shape) == 2:  # If the image is 2D (h, w), expand to (h, w, 1) or (1, h, w)
            if self.image_format == ImageFormat.HWC:
                self.image = np.expand_dims(self.image, axis=-1)
            elif self.image_format == ImageFormat.CHW:
                self.image = np.expand_dims(self.image, axis=0)

        # Update the color mode
        self.color_mode = new_color_mode
        
    
    def copy(self) -> 'Image':
        """
        Create a deep copy of the current Image instance.

        Returns:
            A new Image instance with the same attributes as the current one.
        """
        # Create a new Image instance with the same image data, color mode, and format
        new_image = Image(
            image_input=copy.deepcopy(self.image),
            color_mode= self.color_mode
            )
        return new_image


    # The remaining methods of the class remain unchanged
    def get_shape(self,format : ImageFormat = ImageFormat.HWC ) -> tuple:
        """Get the shape of the image as a tuple (height, width, channels)."""
        if format == ImageFormat.HWC:
            return self.image.shape
        elif format == ImageFormat.CHW:
            return self.image.shape[::-1]
                 
    def get_width(self) -> int:
        """Get the width of the image."""
        return self.image.shape[1]
        
    def get_height(self) -> int:
        """Get the height of the image."""
        return self.image.shape[0]
    

    def get_number_channel(self) -> int:
        """Get the number of channels in the image."""
        return self.image.shape[2]

    def get_color_mode(self) -> ColorMode:
        """Get the color mode of the image."""
        return self.color_mode

    def get_data(self,format : ImageFormat = ImageFormat.HWC) -> np.ndarray:
        """Get the image as a NumPy array."""
        if format == ImageFormat.HWC:
            return self.image   
        elif format == ImageFormat.CHW: 
            return self.convert_image_format(self.image, ImageFormat.CHW)

    def resize(self, new_size: tuple) -> 'Image':
        """
        Resize the image to the specified dimensions while maintaining 3 dimensions.

        Args:
            new_size: A tuple (width, height) representing the new dimensions.

        Returns:
            self: The Image instance with the resized image.
        """
        # Resize the image using OpenCV
        self.image = cv2.resize(self.image, new_size)

        # Ensure the image has 3 dimensions
        if len(self.image.shape) == 2:  # If the image is 2D (h, w), expand to (h, w, 1)
            self.image = np.expand_dims(self.image, axis=-1)
        elif len(self.image.shape) == 3 and self.image.shape[2] == 1:  # If already (h, w, 1), do nothing
            pass
        elif len(self.image.shape) == 3 and self.image.shape[2] == 3:  # If already (h, w, 3), do nothing
            pass
        else:
            # Handle unexpected shapes (e.g., 4D or invalid)
            raise ValueError(f"Unexpected image shape after resizing: {self.image.shape}")

        return self

    def crop(self, crop_window: dict) -> 'Image':
        """Crop the image using the specified window."""
        self.image = self.image[crop_window['y_min']:crop_window['y_max'],
                                crop_window['x_min']:crop_window['x_max']]
        return self

    def invert_pixels(self) -> 'Image':
        """Invert the pixel values of the image."""
        self.image = cv2.bitwise_not(self.image)
        if len(self.image.shape) == 3 and self.image.shape[2] == 1:
            self.image = np.expand_dims(self.image, axis=-1)
        return self

    def save_raw(self, dest_path: str):
        """Save the image data as a raw binary file."""
        with open(dest_path, 'wb') as f:
            f.write(self.image.tobytes())

    def save_jpg(self, dest_path: str):
        """Save the image as a JPG file."""
        if len(self.image.shape) == 3 and self.image.shape[2] == 3:
            image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        else:
            image = self.image
        cv2.imwrite(dest_path, image)

    def show(self, window_name: str = "Image", key: int = ord('q')):
        """Display the image in a window. Close the window when the specified key is pressed."""
        cv2.imshow(window_name, self.image)
        while True:
            key_pressed = cv2.waitKey(1) & 0xFF  # Wait for 1 ms and check the key pressed
            if key_pressed == key or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()

    @classmethod
    def convert_image_format(cls, image: np.ndarray, target_format: ImageFormat) -> np.ndarray:
        """Convert an image between HWC and CHW formats."""
        if len(image.shape) not in {2, 3}:
            raise ValueError(f"Invalid image shape: {image.shape}. Expected 2D (H, W) or 3D (H, W, C).")

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        if target_format == ImageFormat.CHW:
            if len(image.shape) == 3 and image.shape[2] <= 4:
                return np.transpose(image, (2, 0, 1))
            elif len(image.shape) == 3 and image.shape[0] <= 4:
                return image
        elif target_format == ImageFormat.HWC:
            if len(image.shape) == 3 and image.shape[0] <= 4:
                return np.transpose(image, (1, 2, 0))
            elif len(image.shape) == 3 and image.shape[2] <= 4:
                return image
        else:
            raise ValueError(f"Unsupported target format: {target_format}")

        raise ValueError(f"Cannot convert image with shape {image.shape} to {target_format} format.")

    @classmethod
    def jpg_to_raw(cls, input_file: str, grayscale: bool = False) -> np.ndarray:
        """Convert a JPEG image to a raw array."""
        if grayscale:
            img_data = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
        else:
            img_data = cv2.imread(input_file)
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        return img_data

    @classmethod
    def resize_image(cls, image: np.ndarray, new_size: tuple) -> np.ndarray:
        """Resize an image using OpenCV."""
        return cv2.resize(image, new_size)

    @classmethod
    def crop_image(cls, image: np.ndarray, crop_window: dict) -> np.ndarray:
        """Crop an image according to the specified window."""
        return image[crop_window['y_min']:crop_window['y_max'],
                     crop_window['x_min']:crop_window['x_max']]

    @classmethod
    def save_raw_image(cls, image: np.ndarray, dest_path: str):
        """Save raw image data to file."""
        with open(dest_path, 'wb') as f:
            f.write(image.tobytes())

    @classmethod
    def save_jpg_image(cls, image: np.ndarray, dest_path: str):
        """Save an image as JPG using OpenCV."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dest_path, image)

    @classmethod
    def invert_pixel(cls, img: np.ndarray) -> np.ndarray:
        """Invert the pixel values of a uint8 image."""
        inverted_img = cv2.bitwise_not(img)
        if len(img.shape) == 3 and img.shape[2] == 1:
            inverted_img = np.expand_dims(inverted_img, axis=-1)
        return inverted_img
    
    @classmethod
    def pad_image(cls, image: np.ndarray, target_h: int, target_w: int, image_format: ImageFormat = ImageFormat.HWC) -> np.ndarray:
        """Add padding if image dimensions are smaller than target size.

        Args:
            image: Input image in CHW or HWC format.
            target_h: Target height.
            target_w: Target width.
            image_format: Input image format (CHW or HWC).

        Returns:
            np.ndarray: Padded tensor matching target size, or original if no padding needed.
                        Output maintains the specified format and float32 precision.
        """
        # Convert to CHW format for consistent processing
        if image_format == ImageFormat.HWC:
            image = cls.convert_image_format(image, ImageFormat.CHW)

        # Extract dimensions
        c, h, w = image.shape

        # Return original image if no padding needed
        if h >= target_h and w >= target_w:
            if image_format == ImageFormat.HWC:
                image = cls.convert_image_format(image, ImageFormat.HWC)
            return image

        # Add padding only if necessary
        padded_img = 114 * np.ones((c, target_h, target_w), dtype=np.float32)
        padded_img[:, :h, :w] = image

        # Convert back to the original format
        if image_format == ImageFormat.HWC:
            padded_img = cls.convert_image_format(padded_img, ImageFormat.HWC)

        return padded_img