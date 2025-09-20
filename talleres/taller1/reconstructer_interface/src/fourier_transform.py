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
