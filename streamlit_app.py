import streamlit as st
import requests

st.set_page_config(page_title="EN ↔ KO Translator")

# Hugging Face Inference API URLs
API_URLS = {
    "English → Korean": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-ko",
    "Korean → English": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-ko-en",
}



st.title("English ↔ Korean Translator")

direction = st.radio("Select translation direction", ["English → Korean", "Korean → English"])
text = st.text_area("Enter your text:")

def query(payload, direction):
    response = requests.post(API_URLS[direction], json={"inputs": payload})
    if response.status_code == 200:
        return response.json()[0]["translation_text"]
    else:
        return f"Error: {response.status_code} — Try again later."

if st.button("Translate"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = query(text, direction)
        st.success(result)
