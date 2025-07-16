import streamlit as st
import pandas as pd
import time
from datetime import datetime
import plotly.express as px

# Page config
st.set_page_config(
    page_title="🧠 AI Research Assistant | Daniel Ojeda Rosales",
    page_icon="🤖",
    layout="wide"
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(45deg, #00d4ff, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .author-header {
        font-size: 1.2rem;
        color: #00d4ff;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Header
st.markdown('<h1 class="main-header">🧠 AI Research Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="author-header">👨‍💻 Developed by Daniel Ojeda Rosales</p>', unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3 = st.tabs(["🤖 Chat", "📚 Documents", "📊 Analytics"])

with tab1:
    st.markdown("### 💬 Chat with AI")
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about research papers..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            response = f"Thanks for asking about: {prompt}. This is where your RAG system will respond!"
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    st.markdown("### 📚 Document Upload")
    uploaded_files = st.file_uploader("Upload research papers", type=['pdf', 'txt'], accept_multiple_files=True)
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} files!")

with tab3:
    st.markdown("### 📊 Analytics")
    if st.session_state.messages:
        st.metric("Total Messages", len(st.session_state.messages))
    else:
        st.info("No analytics yet. Start chatting!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #4ecdc4;">
    🎓 Analytics Vidya GenAI Pinnacle Program | Capstone Project
</div>
""", unsafe_allow_html=True)