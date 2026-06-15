import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load trained CNN model
model = load_model("fashion_mnist_cnn.h5")

# Fashion MNIST class labels
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Streamlit page settings
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👕",
    layout="centered"
)

# App title
st.title("👕 Fashion Item Classification Using CNN")

st.write(
    "Upload a fashion image and the CNN model will predict the clothing category."
)

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# Prediction section
if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file)

    # Show uploaded image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Convert image to grayscale
    image = image.convert("L")

    # Invert colors (important for Fashion MNIST)
    image = Image.eval(image, lambda x: 255 - x)

    # Resize image
    image = image.resize((28, 28))

    # Convert image to numpy array
    img_array = np.array(image)

    # Normalize image
    img_array = img_array / 255.0

    # Reshape image for CNN model
    img_array = img_array.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(img_array)

    # Predicted class index
    predicted_class = np.argmax(prediction)

    # Confidence score
    confidence = np.max(prediction) * 100

    # Display prediction
    st.success(
        f"Prediction: {class_names[predicted_class]}"
    )

    # Display confidence
    st.info(
        f"Confidence: {confidence:.2f}%"
    )