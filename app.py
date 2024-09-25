import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model from Hugging Face
model_url = "https://huggingface.co/mars-hm/cnn_tumor/resolve/main/cnn_tumor.h5"
model_path = tf.keras.utils.get_file("cnn_tumor.h5", model_url)
model = tf.keras.models.load_model(model_path)

def preprocess_image(image):
    image = image.resize((128, 128))
    test = np.array(image) / 255.0  # Normalize to [0, 1]
    test = np.expand_dims(test, axis=0)
    return test

def predict_tumor(test):
    prediction = model.predict(test)
    if prediction > 0.5:
        return "Tumor Detected"
    else: 
        return "No Tumor Detected"

# Streamlit app
st.title("Brain Tumor Detection using CNN")

uploaded_file = st.file_uploader("Upload an Image:", type="jpg")

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    img_array = preprocess_image(image)
    result = predict_tumor(img_array)
    
    # Display the uploaded image and prediction result
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.info(f"Prediction: {result}")
