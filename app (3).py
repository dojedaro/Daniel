# ===============================================================================
# Research Paper Answer Bot - Streamlit App
# Created by: Daniel Ojeda Rosales
# ===============================================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid
import re
import sys

# Add the directory containing your RAG pipeline code to the Python path
# This assumes your RAG pipeline code is in the same directory or a subdirectory
# If your pipeline is in a different location, adjust the path accordingly
sys.path.append(".") # Assuming pipeline is in the same directory as app.py

# Import your actual RAG pipeline classes from previous tasks
# You might need to adjust the import path based on your file structure
try:
    from __main__ import RAGPipeline # Adjust if your classes are in a different file
    st.success("✅ RAG Pipeline classes imported successfully!")
except ImportError:
    st.error("❌ Could not import RAG Pipeline classes. Make sure your Task 1-7 code is available.")
    # Define dummy classes to prevent errors if import fails
    class RAGPipeline:
        def query(self, question):
            return type('obj', (object,), {
                'answer': f"Error: RAG Pipeline not loaded. Could not answer '{question}'.",
                'sources': [],
                'confidence_score': 0.0,
                'retrieval_method': 'dummy',
                'token_count': {'total_tokens': 0}
            })()


# ============= RAG SYSTEM LOADING =============
@st.cache_resource
def load_rag_system():
    """Load the full RAG system from previous tasks."""
    try:
        # Re-run the previous cells to get the initialized RAG pipeline
        # This is a workaround for Colab; in a real deployment, you'd structure
        # your code to import and initialize the pipeline directly.
        # For Streamlit Cloud, you would import your classes and initialize them here.
        # We'll simulate loading from the completed task results.

        # Assuming task4_results is available in the environment where this script runs
        # In a real Streamlit app, you would import and initialize your components directly
        # For Colab demo purposes, we access global state, which is not ideal for production
        # THIS PART NEEDS TO BE REPLACED WITH ACTUAL IMPORTS AND INITIALIZATION
        # FOR A PRODUCTION STREAMLIT CLOUD DEPLOYMENT.

        # Example of how you *would* initialize in a standalone app:
        # from your_rag_module import SmartRetriever, RAGPipeline, EmbeddingManager, ChromaVectorStore
        # # Load or re-create components
        # embedding_manager = EmbeddingManager()
        # embedding_manager.initialize_huggingface_embeddings() # Or load pre-computed embeddings
        # vector_store = ChromaVectorStore() # Load or re-create your vector store
        # # ... logic to load indexed data into vector_store ...
        # smart_retriever = SmartRetriever(vector_store, embedding_manager, keyword_retriever_instance) # need keyword retriever instance
        # rag_pipeline_instance = RAGPipeline(smart_retriever)

        # --- SIMULATION FOR COLAB DEMO ---
        # Accessing global state from the notebook execution history
        # THIS IS NOT SUITABLE FOR PRODUCTION STREAMLIT DEPLOYMENT
        if 'task4_results' in globals():
             rag_pipeline_instance = globals()['task4_results']['rag_pipeline']
             st.success("✅ RAG Pipeline loaded from global state (Colab demo mode).")
        else:
             st.warning("⚠️ 'task4_results' not found in global state. Initializing RAGPipeline directly.")
             # Placeholder for initializing without previous task results
             # In a real app, you would load necessary data and components here
             rag_pipeline_instance = RAGPipeline(None) # Pass necessary dependencies

        st.success("✅ Full RAG Pipeline loaded successfully!")
        return rag_pipeline_instance
    except Exception as e:
        st.error(f"❌ Could not load RAG Pipeline: {e}")
        return None


# ============= STREAMLIT APP =============
def main():
    st.set_page_config(page_title="🤖 Research Paper Answer Bot | Daniel Ojeda Rosales", page_icon="🧠", layout="wide")

    st.markdown("""<style>
    /* Basic dark theme similar to the notebook example */
    body {
        color: #ffffff;
        background-color: #1a1a1a;
    }
    .stApp {
        background-color: #1a1a1a;
    }
    .main-header { font-size: 3rem; color: #00d4ff; text-align: center; margin-bottom: 1rem; }
    .creator-name { font-size: 1.2rem; color: #ff6b9d; text-align: center; margin-bottom: 2rem; }
    .chat-message { padding: 1rem; border-radius: 10px; margin: 0.5rem 0; }
    .user-message { background-color: #667eea; color: white; }
    .bot-message { background-color: #11998e; color: white; }
    .stTextInput > div > div > input {
        background-color: #333333;
        color: white;
    }
    .stButton>button {
        background-color: #00d4ff;
        color: black;
    }
    /* Hide Streamlit footer */
    footer {visibility: hidden;}
    </style>""", unsafe_allow_html=True)


    st.markdown("""<div class="main-header">🤖 Research Paper Answer Bot</div>
    <div class="creator-name">✨ Created by Daniel Ojeda Rosales ✨</div>""", unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Load the RAG system once
    rag_system = load_rag_system()

    with st.sidebar:
        st.markdown("### 🎯 About")
        st.markdown("""This is an AI-powered research assistant that can answer questions about:\n- 🤖 Transformer architectures\n- 🧠 BERT and GPT models\n- 📖 Deep learning research\n- 🔍 Natural language processing""")
        st.markdown("### ⚙️ Settings")
        response_style = st.selectbox("Response Style", ["Academic", "Professional", "Casual"])
        show_sources = st.checkbox("Show Sources", value=True)

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    st.markdown("## 💬 Ask me about AI/ML research papers!")

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""<div class="chat-message user-message"><strong>👤 You:</strong> {message["content"]}</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="chat-message bot-message"><strong>🤖 Bot:</strong> {message["content"]}</div>""", unsafe_allow_html=True)
            if show_sources and "sources" in message:
                st.markdown("**📚 Sources:**")
                for i, source in enumerate(message["sources"][:3], 1):
                    # Ensure source keys exist before accessing
                    filename = source.get('filename', 'Unknown')
                    page = source.get('page', 'N/A')
                    score = source.get('score', 0.0)
                    st.markdown(f"   {i}. {filename} (Page {page}) - Score: {score:.2f}")


    user_input = st.text_input("💭 Type your question here...", key="user_input")
    col1, col2 = st.columns([1, 4])
    with col1:
        send_button = st.button("🚀 Send", use_container_width=True)

    if send_button and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("🤔 Thinking..."):
            if rag_system:
                response = rag_system.query(user_input)
                bot_message = {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources # Ensure sources are included
                }
            else:
                 # Fallback if RAG system failed to load
                 bot_message = {
                     "role": "assistant",
                     "content": f"I'm sorry, the research bot is currently unavailable. Could not process your request about: {user_input}",
                     "sources": []
                 }
            st.session_state.messages.append(bot_message)
        st.rerun()

    if not st.session_state.messages:
        st.markdown("### 💡 Try asking:")
        sample_questions = [
            "What is the transformer architecture?",
            "How does BERT work?",
            "Explain attention mechanisms",
            "What are the latest developments in LLMs?"
        ]
        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(f"💭 {question}", key=f"sample_{i}"):
                    st.session_state.messages.append({"role": "user", "content": question})
                    with st.spinner("🤔 Thinking..."):
                        if rag_system:
                            response = rag_system.query(question)
                            bot_message = {
                                "role": "assistant",
                                "content": response.answer,
                                "sources": response.sources # Ensure sources are included
                            }
                        else:
                            # Fallback if RAG system failed to load
                            bot_message = {
                                "role": "assistant",
                                "content": f"I'm sorry, the research bot is currently unavailable. Could not process your request about: {question}",
                                "sources": []
                            }
                        st.session_state.messages.append(bot_message)
                    st.rerun()


    st.markdown("---")
    st.markdown("🤖 **Research Paper Answer Bot** - Powered by AI | Created by Daniel Ojeda Rosales")

if __name__ == "__main__":
    main()
