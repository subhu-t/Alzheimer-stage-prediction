import streamlit as st
from PIL import Image
import os
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('alzheimers_model.h5')

def names(number):
    """Map numerical prediction to class names."""
    if number == 0:
        return 'Non Demented'
    elif number == 1:
        return 'Mild Dementia'
    elif number == 2:
        return 'Moderate Dementia'
    elif number == 3:
        return 'Very Mild Dementia'
    else:
        return 'Error in Prediction'

def predic(img_path, model):
    """Predict the stage of Alzheimer's from the given image."""
    img = Image.open(img_path)  # Use img_path as a variable
    x = np.array(img.resize((128, 128)))
    x = x.reshape(1, 128, 128, 3)
    res = model.predict_on_batch(x)
    classification = np.where(res == np.amax(res))[1][0]  # Get the index of the highest probability
    return f"{res[0][classification] * 100:.2f}% Confidence This Is {names(classification)}"

def save_uploaded_image(uploaded_image):
    """Save the uploaded image to the 'uploads' directory."""
    try:
        os.makedirs('uploads', exist_ok=True)  # Create directory if it doesn't exist
        file_path = os.path.join('uploads', uploaded_image.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return None

# Streamlit app setup
st.title('Alzheimer Stage Prediction')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    # Save image in directory
    file_path = save_uploaded_image(uploaded_image)
    if file_path:
        display_image = Image.open(file_path)
        st.image(display_image, caption='Uploaded Image', use_column_width=True)

        # Function call to predict stage
        output = predic(file_path, model)
        st.text(output)
