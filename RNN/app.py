import streamlit as st

st.set_page_config(
    page_title="RNN Sentiment Analysis",
    page_icon="🎬"
)

st.title("🎬 RNN Movie Review Sentiment Analysis")

review = st.text_area(
    "Enter a movie review"
)

if st.button("Analyze"):

    positive_words = [
        "good",
        "great",
        "excellent",
        "amazing",
        "love",
        "wonderful"
    ]

    score = 0

    text = review.lower()

    for word in positive_words:
        if word in text:
            score += 1

    if score > 0:
        st.success("Positive Review")
    else:
        st.error("Negative Review")