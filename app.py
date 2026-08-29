import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠")


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("models/tumor_model.h5")
    with open("models/class_names.txt") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names


model, class_names = load_artifacts()
IMG_SIZE = (128, 128)

st.title("🧠 Brain Tumor MRI Classifier")
st.caption(
    "CNN trained to flag likely tumor presence in an MRI scan. "
    "Research/portfolio demo only — **not a diagnostic tool**."
)

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded scan", use_container_width=True)

    resized = image.resize(IMG_SIZE)
    arr = np.expand_dims(np.array(resized), axis=0)

    pred = model.predict(arr)[0][0]
    predicted_class = class_names[1] if pred > 0.5 else class_names[0]
    confidence = pred if pred > 0.5 else 1 - pred

    st.metric("Prediction", predicted_class, delta=f"{confidence*100:.1f}% confidence")
    st.warning(
        "This model performs binary image classification only. It does not "
        "estimate prognosis, survival time, or recurrence risk — that requires "
        "clinical/longitudinal data and a different kind of model entirely."
    )
