import numpy as np
import scipy.fft

def get_centered_fourier_transform(matrix: np.ndarray) -> np.ndarray:
  """
  Calculates the fourier transform and then shifts it to the center.

  Parameters:
    - matrix (np.ndarray)
  """

  fourier_transform = scipy.fft.fft2(matrix)
  shifted_fourier_transform = scipy.fft.fftshift(fourier_transform)

  return shifted_fourier_transform

def get_inverse_fourier_transform(matrix: np.ndarray) -> np.ndarray:
  """
  Calculates the inverse fourier transform.

  Parameters:
    - matrix (np.ndarray)
  """

  inverse_fourier_transform = scipy.fft.ifft2(matrix)

  return inverse_fourier_transform
