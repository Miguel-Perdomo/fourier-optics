import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def complex_spectrum_visualization(matrix: np.ndarray, percentile: float = 99) -> np.ndarray:
    """
    Computes a better visualization of the Fourier spectrum:
    - Uses logarithmic magnitude.
    - Scales with a percentile to avoid bright spots saturating the display.
    """
    A = np.abs(matrix)
    vmax = np.percentile(A, percentile)
    return A/ vmax   # normaliza a [0,1] para mostrar bien en imshow


def plot_picture(img: np.ndarray, title: str = ""):
    """
    Displays an image (2D array) in grayscale using Streamlit.
    """
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)  # asumimos normalización [0,1]
    if title:
        ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig)


def create_square_patch(coordinates: np.ndarray):
    """
    Creates a matplotlib Rectangle from square coordinates.
    """
    x1, x2, y1, y2 = coordinates
    square_patch = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=1,
        edgecolor='red',
        facecolor='none'
    )
    return square_patch


def create_circle_patch(coordinates: np.ndarray):
    """
    Creates a matplotlib Circle from circle coordinates.
    """
    center_x, center_y, radius = coordinates
    circle_patch = patches.Circle(
        (center_x, center_y),
        radius,
        linewidth=1,
        edgecolor='red',
        facecolor='none'
    )
    return circle_patch
