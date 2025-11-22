import numpy as np

def create_square_mask(img: np.ndarray, sample_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates a square mask centered on the image.

    Parameters:
      - img (np.ndarray): 2D array of the picture.
      - sample_size (int): given sample size.

    Returns:
      - square_mask (np.ndarray): 2D mask with ones inside the square and zeros elsewhere.
      - square_coordinates (np.ndarray): [x1, x2, y1, y2] coordinates of the square vertices.
    """
    h, w = img.shape
    cy, cx = h // 2, w // 2   # center coordinates (integer)

    # Define side length proportional to sampling factor
    square_side_length = h // sample_size  

    half_side = square_side_length // 2
    x1, x2 = int(cx - half_side), int(cx + half_side)
    y1, y2 = int(cy - half_side), int(cy + half_side)

    square_mask = np.zeros_like(img, dtype=np.float32)
    square_mask[y1:y2, x1:x2] = 1.0

    square_coordinates = np.array([x1, x2, y1, y2])
    return square_mask, square_coordinates


def create_circle_mask(img: np.ndarray, sample_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates a circular mask centered on the image.

    Parameters:
      - img (np.ndarray): 2D array of the picture.
      - sample_size (int): given sample size.

    Returns:
      - circle_mask (np.ndarray): 2D mask with ones inside the circle and zeros elsewhere.
      - circle_coordinates (np.ndarray): [cx, cy, radius].
    """
    h, w = img.shape
    cy, cx = h // 2, w // 2   # center coordinates (integer)

    # Radius proportional to sampling factor
    radius = h / (2 * sample_size)

    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2

    circle_mask = np.zeros_like(img, dtype=np.float32)
    circle_mask[mask] = 1.0

  radius = h / (sample_size * sample_size)
  
  circular_mask = np.zeros_like(img)
  Y, X = np.indices((h, w))
  mask = (X - center[0]) ** 2 + (Y - center[1]) ** 2 <= (radius ** 2)
  circular_mask[mask] = 1 

  circle_coordinates = np.array([center[0], center[1], radius])

def filter_data(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a frequency mask (element-wise multiplication)."""
    return data * mask
