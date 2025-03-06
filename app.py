import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import skimage.feature as skf
from skimage import segmentation, color

# Define image dimensions
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Load trained CNN model
MODEL_PATH = "crack_detection_model.h5"
model_cnn = load_model(MODEL_PATH)

# Load second model (feature-based)
FEATURE_MODEL_PATH = "trained_model.h5"
model_feature_based = load_model(FEATURE_MODEL_PATH)

def preprocess_image(image):
    resized_image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    normalized_image = resized_image / 255.0
    reshaped_image = np.expand_dims(normalized_image, axis=0)
    return reshaped_image

def predict_crack_cnn(image):
    processed_image = preprocess_image(image)
    prediction = model_cnn.predict(processed_image)[0]
    return "Crack Detected" if np.argmax(prediction) == 1 else "No Crack"

def graph_based_segmentation(image):
    segments = segmentation.slic(image, compactness=30, n_segments=400)
    segmented_image = color.label2rgb(segments, image, kind='avg')
    return segmented_image, segments

def extract_crack_features(image, segments):
    features = []
    for segment_label in np.unique(segments):
        mask = segments == segment_label
        segment_image = image * mask[:, :, np.newaxis]
        gray_segment = cv2.cvtColor(segment_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        
        glcm = skf.graycomatrix(gray_segment, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = skf.graycoprops(glcm, 'contrast')
        dissimilarity = skf.graycoprops(glcm, 'dissimilarity')
        homogeneity = skf.graycoprops(glcm, 'homogeneity')
        energy = skf.graycoprops(glcm, 'energy')
        correlation = skf.graycoprops(glcm, 'correlation')
        
        edges = cv2.Canny(gray_segment, 50, 150)
        edge_count = np.sum(edges > 0)
        edge_density = edge_count / (gray_segment.shape[0] * gray_segment.shape[1])
        
        features.append([contrast[0, 0], dissimilarity[0, 0], homogeneity[0, 0], energy[0, 0], correlation[0, 0], edge_count, edge_density])
    
    return np.array(features)

def predict_crack_feature_based(image):
    segmented_image, segments = graph_based_segmentation(image)
    features = extract_crack_features(image, segments)
    prediction = model_feature_based.predict(features)
    predicted_label = np.argmax(prediction) if len(prediction.shape) > 1 else prediction[0]
    return "Crack Detected" if predicted_label == 1 else "No Crack"

st.title("Crack Detection Using CNN & Feature-Based Model")
st.write("Upload an image to compare the results of both models.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("### Crack Detection Results:")
    
    cnn_result = predict_crack_cnn(image)
    st.write(f"**CNN Model:** {cnn_result}")

    feature_result = predict_crack_feature_based(image)
    st.write(f"**Image Features based Model:** {feature_result}")

st.write("### Live Crack Detection")

if "live_detection" not in st.session_state:
    st.session_state.live_detection = False

if st.button("Start Live Detection"):
    st.session_state.live_detection = True

if st.session_state.live_detection:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    stop_button = st.button("Stop Detection")
    
    while st.session_state.live_detection:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = predict_crack_cnn(frame)
        color = (0, 255, 0) if result == "No Crack" else (0, 0, 255)
        cv2.putText(frame, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        stframe.image(frame, channels="BGR")
        
        if stop_button:
            st.session_state.live_detection = False  
            break
    
    cap.release()
