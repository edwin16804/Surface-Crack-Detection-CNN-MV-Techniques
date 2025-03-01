import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model = load_model("crack_detection_model.h5")  # Ensure the model is in the same directory

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

    if st.button("Classify"):
        # Preprocess and make prediction
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        
        # Interpret result
        result = "Crack Detected 🚨" if prediction[0][0] > 0.5 else "No Crack ✅"
        confidence = prediction[0][0] * 100 if prediction[0][0] > 0.5 else (1 - prediction[0][0]) * 100
        
        # Display result
        st.write(f"**Prediction:** {result}")
        st.write(f"**Confidence:** {confidence:.2f}%")
