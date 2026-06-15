import streamlit as st
from collections import defaultdict

st.set_page_config(
    page_title="RNN Next Word Predictor",
    page_icon="🔤"
)

st.title("🔤 RNN Next Word Prediction")

st.write(
    "Enter a word and predict the next likely word."
)

corpus = """
machine learning is amazing
machine learning is powerful
deep learning uses neural networks
deep learning is a subset of machine learning
artificial intelligence uses machine learning
neural networks are powerful
"""

words = corpus.lower().split()

model = defaultdict(list)

for i in range(len(words) - 1):
    model[words[i]].append(words[i + 1])

word = st.text_input(
    "Enter a word",
    "machine"
).lower()

if st.button("Predict Next Word"):

    if word in model:

        next_words = model[word]

        freq = {}

        for w in next_words:
            freq[w] = freq.get(w, 0) + 1

        prediction = max(freq, key=freq.get)

        st.success(
            f"Predicted Next Word: {prediction}"
        )

        st.write("Possible Next Words:")

        for w in set(next_words):
            st.write("-", w)

    else:
        st.error("Word not found in vocabulary.")