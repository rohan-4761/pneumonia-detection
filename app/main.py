import os
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/model/our_model.h5"

# Load the model only once
model = tf.keras.models.load_model(model_path)


def predict_pneumonia(image_path, trained_model):
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format
    img = img.resize((224, 224))  # Resize to the target size
    img_array = np.array(img)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_data = preprocess_input(img_array)

    # Make a prediction
    prediction = trained_model.predict(img_data)

    return prediction[0]


# Streamlit App
st.title('Pneumonia Detection')

uploaded_image = st.file_uploader("Upload an x-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_pneumonia(uploaded_image, model)
            if prediction[0] > prediction[1]:
                st.success('Person is safe.')
            else:
                st.error('Person is affected with Pneumonia.')

