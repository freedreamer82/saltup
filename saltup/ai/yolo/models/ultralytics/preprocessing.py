import cv2
import numpy as np

def preprocess(image_path, target_height:int, target_width:int):
    """
    Preprocesses the input image before performing inference.

    Returns:
        image_data: Preprocessed image data ready for inference.
    """
    # Read the input image using OpenCV
    img = cv2.imread(image_path)

    # Get the height and width of the input image
    img_height, img_width = img.shape[:2]
    
    # Resize the image to match the input shape
    img = cv2.resize(img, (target_width, target_height))

    # Normalize the image data by dividing it by 255.0
    image_data = np.array(img) / 255.0
    # Transpose the image to have the channel dimension as the first dimension
    image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

    # Expand the dimensions of the image data to match the expected input shape
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    return image_data