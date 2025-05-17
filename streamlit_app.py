import streamlit as st
import requests

# Load your Hugging Face token from Streamlit secrets
HF_TOKEN = st.secrets["HF_TOKEN"]
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# Correct API endpoints
API_URLS = {
    "English → Korean": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-ko",
    "Korean → English": "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-ko-en"
}

# Translation request
def translate(text, direction):
    url = API_URLS[direction]
    try:
        response = requests.post(url, headers=headers, json={"inputs": text})
        if response.status_code == 200:
            return response.json()[0]["translation_text"]
        else:
            return f"Error: {response.status_code} — Try again later."
    except Exception as e:
        return f"❌ Exception: {e}"

# Streamlit UI
st.set_page_config(page_title="English ↔ Korean Translator")
st.title("English ↔ Korean Translator")

direction = st.radio("Select translation direction", ["English → Korean", "Korean → English"])
text = st.text_area("Enter your text:")
if st.button("Translate"):
    if text.strip():
        result = translate(text, direction)
        st.success(result)
