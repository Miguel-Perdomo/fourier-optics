import numpy as np
import cv2

def upload_gray_image(path: str) -> np.ndarray:
    """
    Uploads image into an array.

    Parameters:
        - path (str): image path
    Returns:
        - image (np.ndarray): array that contains the image
    """
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image

def take_sample(img: np.ndarray, sample_size: int) -> np.ndarray:
    """
    Takes a sample of pixels of a picture by setting to 0 
    the picture array every sample_size rows and columns.

    Parameters:
    - img (np.ndarray): 2d array of the picture.
    - sample_size (int): given sample size.

    Returns:
        sampled array

    """
    sample_mask = np.zeros_like(img)
    sample_mask[::sample_size, ::sample_size] = 1
    img_sample= img * sample_mask
    return img_sample
