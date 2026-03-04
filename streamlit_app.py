import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import psutil
import os

# --- MONITORING UTILITY ---
def get_system_usage():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent(interval=0.1)
    return cpu_percent, mem_mb

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Biomedical Image Analysis Monitor", layout="wide")

# --- INSTRUCTIONS SECTION ---
with st.expander("📖 Instructions & Learning Objectives", expanded=True):
    st.markdown("""
    **Goal:** Explore traditional image processing techniques used in biomedical imaging.
    1. **Compare Algorithms:** Observe how Sobel handles noise vs. how Canny suppresses it.
    2. **Parameter Tuning:** Adjust thresholds to see how edge sensitivity changes.
    3. **Segmentation:** Use Otsu's method to automatically separate foreground from background.
    **Note:** Gaussian Blur is applied before Otsu's method to improve result stability.
    """)

# Sidebar for parameters
st.sidebar.header("Processing Parameters")
mode = st.sidebar.radio(
    "Choose a mode:",
    ("Synthetic Image", "Upload Image"),
    help="Select 'Synthetic' for the notebook rectangle or 'Upload' for real-world data."
)

noise_level = st.sidebar.slider("Noise Level", 0, 100, 50, help="Higher values add more random intensity to the image pixels.")
kernel_size = st.sidebar.slider("Sobel Kernel Size", 3, 11, 5, 2, help="Larger kernels average over more pixels, reducing noise but blurring edges.")
threshold1 = st.sidebar.slider("Canny Threshold 1", 0, 255, 100, help="The lower bound for edge linking.")
threshold2 = st.sidebar.slider("Canny Threshold 2", 0, 255, 200, help="The upper bound for initial edge detection.")
blur_sigma = st.sidebar.slider("Gaussian Blur Sigma", 0.0, 5.0, 1.0, help="Standard deviation for Gaussian kernel. Essential for Otsu's method.")
colormap = st.sidebar.selectbox("Colormap", ["gray", "viridis", "plasma", "magma", "inferno"], help="Changes the visual representation of intensity values.")

# --- PERFORMANCE METRICS ---
st.sidebar.markdown("---")
st.sidebar.subheader("App Performance")
cpu, mem = get_system_usage()
col_cpu, col_mem = st.sidebar.columns(2)
col_cpu.metric("CPU Usage", f"{cpu}%")
col_mem.metric("Memory", f"{mem:.1f} MB")

# --- MAIN LOGIC ---
st.warning("Please do not upload any sensitive or personal data.")

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img = np.array(image)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        st.info("Please upload an image to continue.")
        img = None
else:
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (70, 70), 255, -1)
    noise = np.random.randint(0, noise_level, (100, 100), dtype=np.uint8)
    img = cv2.add(img, noise)

if img is not None:
    # 1. Edge Detection
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    canny_edges = cv2.Canny(img, threshold1, threshold2)

    # 2. Otsu's Thresholding (Notebook Problem 3.2)
    # Apply blur first as recommended in the notebook
    blurred = cv2.GaussianBlur(img, (5, 5), blur_sigma)
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Normalize for display
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Display 4-Column Layout
    st.header("Analysis Results")
    cols = st.columns(4)
    
    titles = ["Original/Noisy", "Sobel Filter", "Canny Edge", "Otsu Threshold"]
    images = [img, sobel_edges, canny_edges, otsu_thresh]
    tooltips = [
        "The raw input image with noise.",
        "Gradient-based detection; sensitive to noise.",
        "Multi-stage algorithm with noise suppression.",
        "Binarization that maximizes inter-class variance."
    ]

    for i, col in enumerate(cols):
        with col:
            st.subheader(titles[i], help=tooltips[i])
            fig, ax = plt.subplots()
            ax.imshow(images[i], cmap=colormap)
            ax.axis('off')
            st.pyplot(fig)
