import streamlit as st
import numpy as np

st.set_page_config(
    page_title="ANN Demo",
    page_icon="🧠"
)

st.title("🧠 Artificial Neural Network Demo")

st.write(
    "Simple ANN forward propagation demonstration."
)

x1 = st.slider("Input X1", 0.0, 10.0, 5.0)
x2 = st.slider("Input X2", 0.0, 10.0, 5.0)

w11, w12 = 0.8, 0.4
w21, w22 = 0.3, 0.9

h1 = max(0, x1*w11 + x2*w21)
h2 = max(0, x1*w12 + x2*w22)

output = 1/(1 + np.exp(-(0.7*h1 + 0.5*h2)))

st.subheader("Hidden Layer")

st.write("Neuron 1:", round(h1, 3))
st.write("Neuron 2:", round(h2, 3))

st.subheader("Output Layer")

st.metric(
    "Prediction Probability",
    f"{output:.2%}"
)

if output > 0.5:
    st.success("Class 1")
else:
    st.error("Class 0")