from pathlib import Path

import numpy as np
import streamlit as st
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.tflite"
CLASS_NAMES = ["apple", "banana", "orange"]
IMAGE_SIZE = (32, 32)


st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="centered")


@st.cache_resource
def load_interpreter():
    """Load and initialise the TensorFlow Lite model once per app session."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    return interpreter


def prepare_image(image: Image.Image, input_dtype: np.dtype) -> np.ndarray:
    """Convert an uploaded image into the model's (1, 32, 32, 3) input tensor."""
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)

    # The model contains its own Rescaling(1./255) layer, so retain 0-255 pixels.
    if np.issubdtype(input_dtype, np.integer):
        array = array.astype(input_dtype)
    else:
        array = array.astype(np.float32)

    return np.expand_dims(array, axis=0)


def predict(image: Image.Image) -> np.ndarray:
    interpreter = load_interpreter()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_tensor = prepare_image(image, input_details["dtype"])
    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details["index"])[0]

    # The final Dense layer returns logits, so convert them to probabilities.
    logits = logits - np.max(logits)
    return np.exp(logits) / np.exp(logits).sum()


st.title("🍎 Fruit Multi-Class Classifier")
st.write(
    "Upload a fruit image to classify it as an **apple**, **banana**, or **orange**."
)

with st.sidebar:
    st.header("Model details")
    st.write("**Architecture:** 3-layer CNN")
    st.write("**Input:** 32 × 32 RGB")
    st.write("**Classes:** Apple, Banana, Orange")
    st.write("**Reported test accuracy:** 96.15%")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Upload a JPG or PNG image to get started.")
else:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Classify fruit", type="primary"):
        try:
            with st.spinner("Classifying image..."):
                probabilities = predict(image)

            predicted_index = int(np.argmax(probabilities))
            predicted_label = CLASS_NAMES[predicted_index].title()
            confidence = probabilities[predicted_index]

            st.success(
                f"Prediction: **{predicted_label}** ({confidence:.1%} confidence)"
            )
            st.subheader("Class probabilities")
            st.bar_chart(
                {
                    label.title(): float(score)
                    for label, score in zip(CLASS_NAMES, probabilities)
                }
            )
        except Exception as error:
            st.error(f"Could not run the prediction: {error}")
