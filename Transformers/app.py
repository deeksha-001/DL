import streamlit as st
from utils.translator import translate_text

st.set_page_config(
    page_title="AI Language Translator",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 AI Language Translator")

st.write(
    "Translate English text into French or Telugu"
)

language = st.selectbox(
    "Select Target Language",
    ["French", "Telugu"]
)

text = st.text_area(
    "Enter English Text",
    height=200
)

if st.button("Translate"):

    if text.strip() == "":
        st.warning(
            "Please enter text"
        )

    else:

        with st.spinner(
            "Translating..."
        ):

            result = translate_text(
                text,
                language
            )

        st.success(
            "Translation Complete"
        )

        st.subheader(
            "Translated Text"
        )

        st.write(result)