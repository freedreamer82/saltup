import numpy as np
import PIL
import os
import cv2
from PIL import Image


def load_grayscale_image(image_path)->PIL.Image:
    im = Image.open(image_path)
    image = im.convert('L')
    return image
   
def jpg_to_raw_array(input_file: str, grayscale : bool = False ) -> np.ndarray:
    '''
    Conversion into raw format of given JPEG image as given from 'path'.
    Args:
        - path: str variable where the image of interested is located.
    Return:
        - img_raw_pixels: np.ndarray file i.e. the raw file
    '''

    # Apri l'immagine JPEG usando Pillow
    with Image.open(input_file) as img:
        # Converti l'immagine in formato RGB se non lo è già
        if not grayscale :
            img = img.convert('RGB')
        else :
            img = img.convert('L')

        img_data = np.array(img)


    return img_data

def resize_image(image:PIL.Image, new_size:tuple)->PIL.Image:

    im1 = image.resize(new_size)
        
    return im1
    
    
def crop_image(image:PIL.Image, crop_window:dict)->PIL.Image:
    """
    Crops the input image according to the specified window.

    Args:
        image (PIL.Image): The input image to be cropped.
        crop_window (dict): A dictionary specifying the cropping window with the following keys:
            - 'x_min' (int): The starting x-coordinate of the crop window.
            - 'y_min' (int): The starting y-coordinate of the crop window.
            - 'x_max' (int): The ending x-coordinate of the crop window.
            - 'y_max' (int): The ending y-coordinate of the crop window.

    Returns:
        PIL.Image: The cropped image.
    """
    
    new_image = image.crop([crop_window['x_min'],crop_window['y_min'], crop_window['x_max'], crop_window['y_max']])
     
    return new_image

def save_raw_image(image:PIL.Image, dest_path:str):
    
    output = os.path.join(dest_path)
        
    with open(output, 'wb') as f:
        f.write(image.tobytes())
        
def save_jpg_image(image:PIL.Image, dest_path:str):
    image.save(dest_path)
    
def invert_pixel(img:np.ndarray) -> np.ndarray:
    """Invert the pixel of a uint8 image

    Args:
        img (np.ndarray): an array of uint8 image

    Returns:
        np.ndarray: the pixle inverted array image
    """
    img = np.max(img) - img
    return img

