import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="ANN Classifier", page_icon="🧠")

st.title("🧠 ANN Breast Cancer Prediction")

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))

st.success(f"Model Accuracy: {acc:.2%}")

st.subheader("Enter Feature Values")

inputs = []

for i, feature in enumerate(data.feature_names):
    val = st.number_input(
        feature,
        value=float(X.iloc[:, i].mean())
    )
    inputs.append(val)

if st.button("Predict"):

    sample = scaler.transform([inputs])

    prediction = model.predict(sample)[0]

    if prediction == 1:
        st.success("Prediction: Benign")
    else:
        st.error("Prediction: Malignant")