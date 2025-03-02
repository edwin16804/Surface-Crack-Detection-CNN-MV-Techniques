import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from skimage import segmentation, color
from sklearn.preprocessing import StandardScaler
import skimage.feature as skf
from PIL import Image

# Load trained model
model = load_model("trained_model.h5")  # Ensure the model file is in the same directory

def graph_based_segmentation(image):
    # Convert PIL image to NumPy array
    image = np.array(image)
    
    # Apply graph-based segmentation
    segments = segmentation.slic(image, compactness=30, n_segments=400)
    segmented_image = color.label2rgb(segments, image, kind='avg')
    return segmented_image, segments

def extract_crack_features(image, segments):
    features = []
    for segment_label in np.unique(segments):
        mask = segments == segment_label
        segment_image = image * mask[:, :, np.newaxis]
        gray_segment = cv2.cvtColor(segment_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        
        # Calculate GLCM features
        glcm = skf.graycomatrix(gray_segment, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = skf.graycoprops(glcm, 'contrast')
        dissimilarity = skf.graycoprops(glcm, 'dissimilarity')
        homogeneity = skf.graycoprops(glcm, 'homogeneity')
        energy = skf.graycoprops(glcm, 'energy')
        correlation = skf.graycoprops(glcm, 'correlation')
        
        # Calculate edge count using Canny edge detection
        edges = cv2.Canny(gray_segment, threshold1=50, threshold2=150)  # Adjust thresholds for crack detection
        edge_count = np.sum(edges > 0)  # Count the number of edge pixels
        
        # Calculate edge density (edges per pixel)
        edge_density = edge_count / (gray_segment.shape[0] * gray_segment.shape[1])
        
        # Append features
        features.append([
            contrast[0, 0], dissimilarity[0, 0], homogeneity[0, 0], energy[0, 0], correlation[0, 0],
            edge_count, edge_density
        ])
    
    return np.array(features)

# Streamlit UI
st.title("Surface Crack Detection 🚧")
st.write("Upload an image to classify it as **Crack** or **No Crack**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        # Convert PIL image to NumPy array
        image_np = np.array(image)
        
        # Perform graph-based segmentation
        segmented_image, segments = graph_based_segmentation(image_np)
        
        # Extract GLCM features
        features = extract_crack_features(segmented_image, segments)

        # Average features across segments
        avg_features = np.mean(features, axis=0)

        # Reshape features to match the model's input shape
        data = np.expand_dims(avg_features, axis=0)

        # Debug: Print the shape of the extracted features
        st.write(f"Extracted features shape: {data.shape}")

        # Standardize the features (if required)
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(data)

        # Predict using trained model
        prediction = model.predict(standardized_data)
        st.write(prediction)
        
        # Interpret result
        result = "Crack Detected 🚨" if prediction[0][0] > 0.5 else "No Crack ✅"
        confidence = prediction[0][0] * 100 if prediction[0][0] > 0.5 else (1 - prediction[0][0]) * 100
        
        # Display result
        st.write(f"**Prediction:** {result}")
        st.write(f"**Confidence:** {confidence:.2f}%")