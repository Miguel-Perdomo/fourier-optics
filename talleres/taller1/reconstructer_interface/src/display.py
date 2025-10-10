import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Callable, Tuple

def complex_spectrum_visualization(matrix: np.ndarray):
    return np.abs(matrix)

def plot_picture(img: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot an image in grayscale without axes.

    Parameters:
      - img (np.ndarray): array representing the image.
    
    Returns:
      - fig (matplotlib.figure)
      - ax (matplotlib.axes)

    """
    fig, ax = plt.subplots()
    ax.imshow(img, cmap = "gray")
    ax.axis("off")

    return fig, ax

def create_square_patch(coordinates:np.ndarray):
    """
    Gives a square patch for plotting.

    parameters:
    
      - coordinates (np.ndarray): a 1d array-like with four 
        values [x1, x2, y1, y2], defining the square/rectangle
        region in image coordinates.

    returns:
      - square_patch (matplotlib.figure): an square patch that
        can be plotted.

    """
    x1, x2, y1, y2 = coordinates

    square_patch = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=1,
        edgecolor="white",
        facecolor="none"
    )

    return square_patch

def create_circle_patch(coordinates: np.ndarray):
    """
    Gives a circular patch for plotting.

    parameters:
    
      - coordinates (np.ndarray): a 1d array-like with four 
        values [center_x, center_y, radius], defining the
        circular region in image coordinates.

    returns:
      - circle_patch (matplotlib.figure): a circular patch 
        that can be plotted.

    """
    center_x, center_y, radius = coordinates

    circle_patch = patches.Circle(
      (center_x, center_y),
      radius,
      linewidth=1,
      edgecolor='white',
      facecolor='none' 
    )

    return circle_patch
