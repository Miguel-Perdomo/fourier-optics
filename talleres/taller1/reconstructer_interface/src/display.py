import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def complex_spectrum_visualization(matrix: np.ndarray):
    return np.abs(matrix)

def plot_picture(img: np.ndarray):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap = "gray")
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
    edgecolor='white',
    facecolor='none' 
  )

  return square_patch

def create_circle_patch(coordinates: np.ndarray):
  """
  Creates a matplotlib Circle from square coordinates.
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