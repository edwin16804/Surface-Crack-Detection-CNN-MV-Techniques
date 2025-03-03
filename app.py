import os
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import load_model

# Define image dimensions
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Load trained model
MODEL_PATH = "crack_detection_model.h5"
model = load_model(MODEL_PATH)

def preprocess_image(image):
    resized_image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    normalized_image = resized_image / 255.0
    reshaped_image = np.expand_dims(normalized_image, axis=0)
    return reshaped_image

def predict_crack(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]
    return "Crack Detected" if np.argmax(prediction) == 1 else "No Crack"

# Streamlit App
st.title("Crack Detection Using CNN")
st.write("Upload an image or use live detection to detect cracks.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("### Crack Detection Result:")
    result = predict_crack(image)
    st.write(f"**{result}**")

# Live Detection using Webcam
st.write("### Live Crack Detection")

# Initialize session state for live detection
if "live_detection" not in st.session_state:
    st.session_state.live_detection = False

if st.button("Start Live Detection"):
    st.session_state.live_detection = True

if st.session_state.live_detection:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    stop_button = st.button("Stop Detection")  # Placed outside loop

    while st.session_state.live_detection:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = predict_crack(frame)
        color = (0, 255, 0) if result == "No Crack" else (0, 0, 255)
        cv2.putText(frame, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        stframe.image(frame, channels="BGR")

        if stop_button:
            st.session_state.live_detection = False  # Stop loop
            break
    
    cap.release()
