import streamlit as st
import requests

# Load Hugging Face token from Streamlit secrets
HF_TOKEN = st.secrets["HF_TOKEN"]

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# New model: Facebook's NLLB-200, supports English ↔ Korean
API_URL = "https://api-inference.huggingface.co/models/facebook/nllb-200-3.3B"

# Translation request using src and tgt language codes
def translate(text, direction):
    src_lang = "eng_Latn" if "English" in direction else "kor_Hang"
    tgt_lang = "kor_Hang" if "Korean" in direction else "eng_Latn"

    payload = {
        "inputs": text,
        "parameters": {
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["translation_text"]
    else:
        return f"Error: {response.status_code} — Try again later."

# Streamlit UI
st.set_page_config(page_title="English ↔ Korean Translator")
st.title("English ↔ Korean Translator")

direction = st.radio("Select translation direction", ["English → Korean", "Korean → English"])
text = st.text_area("Enter your text:")
if st.button("Translate"):
    if text.strip():
        result = translate(text, direction)
        st.success(result)
