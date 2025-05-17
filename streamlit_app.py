import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="EN ↔ KO Translator")

en2ko = pipeline("translation_en_to_ko", model="Helsinki-NLP/opus-mt-en-ko")
ko2en = pipeline("translation_ko_to_en", model="Helsinki-NLP/opus-mt-ko-en")

st.title("English ↔ Korean Translator")

direction = st.radio("Select translation direction", ["English → Korean", "Korean → English"])
text = st.text_area("Enter your text:")

if st.button("Translate"):
    if direction == "English → Korean":
        translation = en2ko(text)[0]['translation_text']
    else:
        translation = ko2en(text)[0]['translation_text']
    st.success(translation)
