import streamlit as st
import cv2
import numpy as np
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import tempfile
from src import utils
from src import display
from src import fourier_transform
from src import filters

# Page configuration

st.set_page_config(page_title="Reconstruction of a sampled image", layout="centered")

# Title

st.markdown(
    """
    <h1 style='text-align: center; color: black;'>
         App de Procesamiento de Imágenes
    </h1>
    <h3 style='text-align: center; color: gray;'>
        Filtros con Transformada de Fourier y Convolución
    </h3>
    """,
    unsafe_allow_html=True,
)

st.write("---")

# Upload picture

uploaded_file = st.file_uploader("Upload a picture", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read picture
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    img = utils.upload_gray_image(tmp_path)

    st.subheader("Sample selection")
    sample = st.slider("Sample size", min_value=1, max_value=10, value=5, step=1)

    img_sample = utils.take_sample(img, sample)

    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original image")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("Sampled image")
        st.image(img_sample, use_container_width=True)

    method = st.radio("Select the method", ["Fourier transform", "Convolution"])

    if method == "Fourier transform":       
        st.header("Method 01: Fourier transform")
        st.write("1. Calculate the Fourier transform of the image.")
        st.subheader("Espectro")
        fourier_transform_img= fourier_transform.get_centered_fourier_transform(img_sample)
        spectrum = display.complex_spectrum_visualization(fourier_transform_img)
        display.plot_picture(spectrum)
        
        filter = st.radio("Select the filter", ["Square", "Circle"])
        if filter == "Square":
            st.subheader("Square filter")

            square_mask, square_coordinates = filters.create_square_mask(img_sample, sample)

            filtered_data = filters.filter_data(fourier_transform_img, square_mask)
            fig, ax = plt.subplots()
            ax.imshow(np.abs(filtered_data), cmap = "gray")
            ax.axis("off")
            square = display.create_square_patch(square_coordinates)
            ax.add_patch(square)
            st.pyplot(fig)
        else:
            st.write("In progress")
            
    else: 
        st.write("In progress")