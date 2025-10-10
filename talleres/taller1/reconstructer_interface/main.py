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
         Reconstruction of sampled images
    </h1>
    <h3 style='text-align: center; color: gray;'>
        Fourier transform and convolution
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

        filter = st.radio("Select the filter", ["Square", "Circle"])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Spectrum")
            fourier_transform_img= fourier_transform.get_centered_fourier_transform(img_sample)
            spectrum = display.complex_spectrum_visualization(fourier_transform_img)
            spectrum_fig, spectrum_ax = display.plot_picture(np.log(1 + spectrum))
            st.pyplot(spectrum_fig)

        with col2: 

            if filter == "Square":
                st.subheader("Square filter")

                square_mask, square_coordinates = filters.create_square_mask(img_sample, sample)

                filtered_data = filters.filter_data(fourier_transform_img, square_mask)
                filtered_data_fig, filtered_data_ax = display.plot_picture(
                                        np.log(1 + display.complex_spectrum_visualization(filtered_data)))
                filtered_data_ax.add_patch(display.create_square_patch(square_coordinates))
                st.pyplot(filtered_data_fig)

            else:
                st.subheader("Circular filter")

                circle_mask, circle_coordinates = filters.create_circle_mask(img_sample, sample)

                filtered_data = filters.filter_data(fourier_transform_img, circle_mask)
                filtered_data_fig, filtered_data_ax = display.plot_picture(
                    np.log(1 + display.complex_spectrum_visualization(filtered_data)))
                filtered_data_ax.add_patch(display.create_circle_patch(circle_coordinates))
                st.pyplot(filtered_data_fig)

        st.header("Reconstruction")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original image")
            st.image(img, use_container_width=True)
        with col2:
            st.subheader("Reconstructed image")
            reconstructed = fourier_transform.get_inverse_fourier_transform(filtered_data)
            reconstructed_fig, reconstructed_ax = display.plot_picture(
                                                    display.complex_spectrum_visualization(reconstructed))
                                                    
            st.pyplot(reconstructed_fig)
            
    else: 
        st.write("In progress")