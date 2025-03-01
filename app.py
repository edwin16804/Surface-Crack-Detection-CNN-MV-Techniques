import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
import numpy as np
import pandas as pd
import skimage.feature as skf
from PIL import Image

# Load trained model
model = load_model("crack_detection_model.h5")  # Ensure the model is in the same directory


def GraphSegmentation(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display

    # Define mask for GrabCut
    row, col, _ = image.shape
    mask = np.zeros((row, col), np.uint8)

    # Initialize models correctly
    bgdModel = np.zeros((1, 65), np.float64)  # Background model
    fgdModel = np.zeros((1, 65), np.float64)  # Foreground model

    x0, y0 = 1, 1  # Ensure valid rectangle coordinates
    x1, y1 = col , row
    rect = (x0, y0, x1 - x0, y1 - y0)

    # Run GrabCut
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 10, cv.GC_INIT_WITH_RECT)

    # Modify mask to extract foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    segmented = image * mask2[:, :, np.newaxis]
    return segmented

def ExtractFeatures(image):

    if len(image.shape) == 3:  
        image = cv2.cv2tColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(image, 50, 150)
    edge_count = np.sum(edges > 0)

    _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask = binary_mask // 255  # Normalize to 0 and 1
    
    crack_pixels = np.sum(binary_mask)
    total_pixels = binary_mask.size
    crack_ratio = crack_pixels / total_pixels

    glcm = skf.graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = skf.graycoprops(glcm, 'contrast')[0, 0]
    correlation = skf.graycoprops(glcm, 'correlation')[0, 0]
    energy = skf.graycoprops(glcm, 'energy')[0, 0]
    homogeneity = skf.graycoprops(glcm, 'homogeneity')[0, 0]
    return [crack_ratio, edge_count, contrast, correlation, energy, homogeneity]





# Define function to preprocess image
def preprocess_image(image):
    image = np.array(image)  # Convert PIL image to NumPy array
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (128, 128))  # Resize to match model input size
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image

# Streamlit UI
st.title("Surface Crack Detection 🚧")
st.write("Upload an image to classify it as **Crack** or **No Crack**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    data=[]

    if st.button("Classify"):
        # Preprocess and make prediction
        segmented = GraphSegmentation(image)
        features = ExtractFeatures(segmented)

        data.append(features)
        prediction = model.predict(data)
        
        # Interpret result
        result = "Crack Detected 🚨" if prediction[0][0] > 0.5 else "No Crack ✅"
        confidence = prediction[0][0] * 100 if prediction[0][0] > 0.5 else (1 - prediction[0][0]) * 100
        
        # Display result
        st.write(f"**Prediction:** {result}")
        st.write(f"**Confidence:** {confidence:.2f}%")
