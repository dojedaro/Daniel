import streamlit as st
import requests

try:
    # your existing imports and logic here
    HF_TOKEN = st.secrets["HF_TOKEN"]
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

except Exception as e:
    st.error("⚠️ App crashed:")
    st.code(traceback.format_exc())
    st.stop()

HF_TOKEN = st.secrets["HF_TOKEN"]

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

API_URLS = {
    "English → Korean": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-ko",
    "Korean → English": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-ko-en",
}

def query(payload, direction):
    response = requests.post(API_URLS[direction], headers=headers, json={"inputs": payload})
    if response.status_code == 200:
        return response.json()[0]["translation_text"]
    else:
        return f"Error: {response.status_code} — Try again later."

# UI
st.title("English ↔ Korean Translator")
direction = st.radio("Select translation direction", ["English → Korean", "Korean → English"])
text = st.text_area("Enter your text:")
if st.button("Translate"):
    if text.strip():
        result = query(text, direction)
        st.success(result)
