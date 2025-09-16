import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.fft

img_path = "/mnt/c/Users/estev/OneDrive - Universidad Nacional de Colombia/Universidad/02_Física/SemestreVIII/ÓpticaDeFourier/FourierOptics_assignments/talleres/taller1/data/foto00_480_480.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

h, w = img.shape

sample = 7

def take_sample(img: np.ndarray, sample_size: int) -> np.ndarray:
    """
    Takes a sample of pixels of a picture by slicing the picture
    array every sample_size rows and columns.

    Parameters:
    - img (np.darray): 2d array of the picture.
    - sample_size (int): given sample size.

    Returns:
        sampled array

    """
    return img[::sample_size, ::sample_size]

img_sample = take_sample(img, sample)

fourier_transform = scipy.fft.fft2(img_sample)
shifted_spectrum = scipy.fft.fftshift(fourier_transform)

visualization = np.abs(shifted_spectrum)

#plt.imshow(visualization, cmap = "gray")
#plt.savefig("resultado.pdf", dpi = 400)
#cv2.imshow("test", visualization)
#cv2.waitKey(0)
#cv2.destroyAllWindows()