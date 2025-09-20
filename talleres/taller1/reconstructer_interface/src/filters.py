import numpy as np

def create_square_mask(img: np.ndarray, sample_size: int) -> np.ndarray: 
  """
  Creates an square mask with side lenght = height / sample * sample.

  Parameters:
    - img (np.darray): 2d array of the picture.
    - sample_size (int): given sample size.

  Returns:
    - square_mask (np.darray): 2d array with ones inside the
    square and zeros everywhere else.
    - square_coordinates (np.darray): array with the coordinates
    of the vertex of the square.

  """
  h, w = img.shape

  center = ((w / 2), (h / 2))

  square_side_length = h / (sample_size * sample_size)

  x1 = int(center[0] - square_side_length)
  x2 = int(center[0] + square_side_length)
  y1 = int(center[1] - square_side_length)
  y2 = int(center[1] + square_side_length)

  square_mask = np.zeros_like(img)
  square_mask[y1:y2, x1:x2] = 1
  
  square_coordinates = np.array([x1, x2, y1, y2])

  return square_mask, square_coordinates

def create_circle_mask(img: np.ndarray, sample_size: int) -> np.ndarray: 
  """
  Creates a circular mask with side radius = height / sample * sample.

  Parameters:
    - img (np.darray): 2d array of the picture.
    - sample_size (int): given sample size.

  Returns:
    - circle_mask (np.darray): 2d array with ones inside the
    circle and zeros everywhere else.
    - circle_coordinates (np.darray): array with the coordinates
    of the center and length of the radius.

  """
  h, w = img.shape

  center = ((w / 2), (h / 2))

  radius = h / (sample_size * sample_size)
  
  circular_mask = np.zeros_like(img)
  Y, X = np.indices((h, w))
  mask = (np.abs(X - center[0]) <= radius) & (np.abs(Y - center[1]) <= radius)
  circular_mask[mask] = 1 

  
  circle_coordinates = np.array([center[0], center[1], radius])

  return circular_mask, circle_coordinates

def filter_data(data: np.ndarray, filter:np.ndarray):
  return data * filter