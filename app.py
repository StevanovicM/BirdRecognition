from tensorflow import _tf_uses_legacy_keras
import keras.models
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from pathlib import Path
import os.path
from keras.models import load_model
from keras.utils import custom_object_scope

image_dir = Path('archive/train')

filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.png'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
class_names = sorted(set(labels))

def F1_score(y_true, y_pred):
    precision = keras.metrics.Precision()(y_true, y_pred)
    recall = keras.metrics.Recall()(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + keras.backend.epsilon()))

MODEL_PATH = 'archive/EfficientNetB0-525-(224 X 224)- 98.97.h5'
with custom_object_scope({'F1_score': F1_score}):
    model = load_model(MODEL_PATH)

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalizing
    image = np.expand_dims(image, axis=0)  # Adding batch dimension
    return image


st.title('Bird Species Prediction')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    if st.button('Predict'):
        # Preprocess the image and prepare it for prediction
        processed_image = preprocess_image(image, target_size=(224, 224))

        # Make a prediction
        prediction = model.predict(processed_image)
        predicted_class = class_names[np.argmax(prediction)]

        # Display the predicted bird species
        st.write(f'Predicted Bird Species: {predicted_class}')

