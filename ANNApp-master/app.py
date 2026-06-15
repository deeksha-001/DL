import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="AI Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

model = load_model("models/ann_model.h5")

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("❤️ AI Heart Disease Risk Assessment System")
st.caption("Artificial Neural Network Based Prediction")

with st.sidebar:

    st.title("🏥 Healthcare AI")

    st.success("ANN Model")

    st.markdown("""
    ### 📊 Parameter Guide

    **Blood Pressure**
    - Normal < 120
    - Elevated 120-139
    - High ≥ 140

    **Cholesterol**
    - Normal < 200
    - Borderline 200-239
    - High ≥ 240
    """)

col1, col2 = st.columns(2)

with col1:

    Age = st.number_input("Age", 1, 100, 40)

    Sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    ChestPainType = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "ASY", "TA"]
    )

    RestingBP = st.number_input(
        "Resting Blood Pressure",
        50,
        250,
        120
    )

    Cholesterol = st.number_input(
        "Cholesterol",
        50,
        700,
        200
    )

with col2:

    FastingBS = st.selectbox(
        "Fasting Blood Sugar",
        [0, 1]
    )

    RestingECG = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    MaxHR = st.number_input(
        "Maximum Heart Rate",
        50,
        250,
        150
    )

    ExerciseAngina = st.selectbox(
        "Exercise Angina",
        ["N", "Y"]
    )

    Oldpeak = st.number_input(
        "Old Peak",
        0.0,
        10.0,
        1.0
    )

    ST_Slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

sex_map = {"Male": 1, "Female": 0}
cp_map = {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3}
ecg_map = {"LVH": 0, "Normal": 1, "ST": 2}
angina_map = {"N": 0, "Y": 1}
slope_map = {"Down": 0, "Flat": 1, "Up": 2}
if st.button("🔍 Analyze Heart Disease Risk"):

    data = np.array([[
        Age,
        sex_map[Sex],
        cp_map[ChestPainType],
        RestingBP,
        Cholesterol,
        FastingBS,
        ecg_map[RestingECG],
        MaxHR,
        angina_map[ExerciseAngina],
        Oldpeak,
        slope_map[ST_Slope]
    ]])

    data = scaler.transform(data)

    prediction = model.predict(
        data,
        verbose=0
    )

    probability = float(prediction[0][0])

    risk = probability * 100

    st.subheader("📊 Risk Analysis")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk,
            title={"text": "Risk %"},
            gauge={
                "axis": {"range": [0, 100]}
            }
        )
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    st.subheader("🩺 Health Status")

    c1, c2, c3, c4 = st.columns(4)

    bp = "Normal" if RestingBP < 120 else "High"
    chol = "Normal" if Cholesterol < 200 else "High"
    sugar = "Normal" if FastingBS == 0 else "High"
    hr = "Normal" if MaxHR >= 100 else "Low"

    c1.metric("BP", bp)
    c2.metric("Cholesterol", chol)
    c3.metric("Blood Sugar", sugar)
    c4.metric("Heart Rate", hr)

    score = 100

    if RestingBP > 140:
        score -= 20

    if Cholesterol > 240:
        score -= 20

    if FastingBS == 1:
        score -= 15

    if ExerciseAngina == "Y":
        score -= 20

    if Age > 60:
        score -= 15

    score = max(score, 0)

    st.subheader("❤️ Health Score")
    st.progress(score)
    st.metric("Health Score", f"{score}/100")

    if risk > 70:
        st.error(f"🔴 HIGH RISK ({risk:.2f}%)")
    elif risk > 40:
        st.warning(f"🟠 MODERATE RISK ({risk:.2f}%)")
    else:
        st.success(f"🟢 LOW RISK ({risk:.2f}%)")

    st.subheader("💡 Recommendations")

    if RestingBP > 140:
        st.write("✅ Monitor blood pressure regularly")

    if Cholesterol > 240:
        st.write("✅ Follow a low cholesterol diet")

    if FastingBS == 1:
        st.write("✅ Reduce sugar intake")

    if ExerciseAngina == "Y":
        st.write("✅ Consult a cardiologist")

    if score >= 80:
        st.write("✅ Overall health status is good")