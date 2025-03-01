import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import skimage.feature as skf
from PIL import Image

# Load trained model
model = load_model("crack_detection_model.h5")  # Ensure the model file is in the same directory

def GraphSegmentation(image):
    image = np.array(image.convert("RGB"))  # Convert PIL image to NumPy array
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

    # Define mask for GrabCut
    row, col, _ = image.shape
    mask = np.zeros((row, col), np.uint8)

    # Initialize models correctly
    bgdModel = np.zeros((1, 65), np.float64)  # Background model
    fgdModel = np.zeros((1, 65), np.float64)  # Foreground model

    rect = (1, 1, col - 2, row - 2)  # Ensure valid rectangle coordinates

    # Run GrabCut
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Modify mask to extract foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    segmented = image * mask2[:, :, np.newaxis]
    
    return segmented

def ExtractFeatures(image):
    """Extract texture and edge features for crack classification."""
    if len(image.shape) == 3:  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

# Streamlit UI
st.title("Surface Crack Detection 🚧")
st.write("Upload an image to classify it as **Crack** or **No Crack**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        segmented = GraphSegmentation(image)
        features = ExtractFeatures(segmented)

        # Convert features to NumPy array for model prediction
        data = np.array(features).reshape(1, -1)
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(data)

        # Predict using trained model
        prediction = model.predict(data)
        st.write(prediction)
        # Interpret result
        result = "Crack Detected 🚨" if prediction[0][0] > 0.5 else "No Crack ✅"
        confidence = prediction[0][0] * 100 if prediction[0][0] > 0.5 else (1 - prediction[0][0]) * 100
        
        # Display result
        st.write(f"**Prediction:** {result}")
        st.write(f"**Confidence:** {confidence:.2f}%")
