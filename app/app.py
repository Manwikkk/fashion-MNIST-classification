import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("trained_fashion_mnist_model.keras") # type: ignore

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title("👕 Fashion MNIST Classifier")
st.write("Upload a grayscale image (28x28) to predict its class.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # convert to grayscale
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    prediction = model.predict(img_array)
    label = np.argmax(prediction)
    
    st.image(image, caption=f"Prediction: {class_names[label]}", use_column_width=True)
    st.write(f"**Predicted Class:** {class_names[label]}")
