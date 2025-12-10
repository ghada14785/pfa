import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import json


# Charger les classes
with open("class_indices.json", "r") as f:
    class_names = json.load(f)
class_names = {int(k): v for k, v in class_names.items()}

# Charger le modèle
model = load_model("plant_disease_prediction_model.h5")

IMG_SIZE = 224  # change selon ton modèle

st.title("🌿 Plant Disease Detection Using CNN")
st.write("Upload an image of a leaf to detect the disease.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

def predict(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    pred = model.predict(img)
    class_id = np.argmax(pred)
    class_name = class_names[class_id]
    
    return class_name

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        result = predict(image)
        st.success(f"🌱 Predicted Disease: **{result}**")

