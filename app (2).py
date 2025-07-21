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
    # --- Replace these imports with your actual class imports ---
    # Example: from your_rag_module import SmartRetriever, RAGPipeline
    # For this example, we'll assume the classes are defined in the main notebook and accessible via sys.path
    # In a real app, ensure these classes are in importable Python files (.py)
    from __main__ import SmartRetriever, RAGPipeline, EmbeddingManager, ChromaVectorStore, KeywordRetriever, RetrievalEvaluator # Assuming all classes were in __main__

    st.success("✅ RAG Pipeline classes imported successfully!")
except ImportError as e:
    st.error(f"❌ Could not import RAG Pipeline classes: {e}. Ensure your Task 1-7 code is in importable .py files.")
    # Define dummy classes to prevent errors if import fails
    class DummyClass: pass
    SmartRetriever = RAGPipeline = EmbeddingManager = ChromaVectorStore = KeywordRetriever = RetrievalEvaluator = DummyClass
    st.warning("⚠️ Using dummy classes. The RAG system will not be functional.")


# ============= RAG SYSTEM LOADING =============
# Use cache_resource to load the potentially heavy RAG components only once
@st.cache_resource
def load_rag_system():
    """Load and initialize the full RAG system."""
    st.info("🔄 Initializing RAG system components (cached)...")
    try:
        # --- IMPORTANT: Adapt this section for your specific RAG pipeline initialization ---
        # In a real deployment, you would load your indexed data and initialize
        # the necessary components here, independent of the Colab notebook's state.

        # Example initialization (replace with your actual loading/init logic):
        # 1. Initialize Embedding Manager
        embedding_manager = EmbeddingManager()
        # You might need to specify model names or load pre-trained models/embeddings here
        embedding_manager.initialize_huggingface_embeddings() # Assuming this method exists and works standalone

        # 2. Initialize Vector Store (and load indexed data)
        vector_store = ChromaVectorStore()
        # --- You need to load your indexed documents/embeddings into the vector_store here ---
        # This might involve reading from persistent storage (e.g., saved Chroma DB, files)
        # Example: vector_store.load_from_disk("path/to/your/chroma_db")
        # Or re-indexing if you have the raw documents:
        # documents, chunks = load_documents(...)
        # vector_store.index_documents(chunks, embedding_manager) # You'd need document loading/chunking logic too

        # --- For this Colab demo, we'll try to access the indexed data if available ---
        # THIS IS NOT SUITABLE FOR PRODUCTION STREAMLIT DEPLOYMENT
        if 'task2_results' in globals():
             st.info("🌐 Attempting to load indexed data from Colab global state...")
             chunks = globals()['task2_results']['chunks']
             # Re-initialize vector store and index for Streamlit environment
             vector_store = ChromaVectorStore()
             collection_name_mini = 'docs_huggingface_all_MiniLM_L6_v2'
             collection_name_mpnet = 'docs_huggingface_all_mpnet_base_v2'
             collection_name_multi_qa = 'docs_huggingface_multi_qa_mpnet_base_cos_v1'

             # Create collections if they don't exist and index
             vector_store.create_collection(collection_name_mini, 'huggingface/all-MiniLM-L6-v2')
             vector_store.create_collection(collection_name_mpnet, 'huggingface/all-mpnet-base-v2')
             vector_store.create_collection(collection_name_multi_qa, 'huggingface/multi-qa-mpnet-base-cos-v1')

             st.info(f"Indexing {len(chunks)} chunks into vector store...")
             vector_store.index_documents(chunks, embedding_manager) # Assuming index_documents handles multiple collections
             st.success("✅ Indexed data loaded and added to vector store.")

        # 3. Initialize Retrievers (they need the vector store and possibly chunks for keyword)
        # You'll need the actual list of documents/chunks for the KeywordRetriever
        # If loading from persistent storage, you'd load chunks here too.
        # For Colab demo, use chunks from global state if available
        if 'chunks' in locals() or 'chunks' in globals():
             keyword_retriever_instance = KeywordRetriever(chunks if 'chunks' in locals() else globals()['chunks'])
             basic_retriever_instance = BasicRetriever(vector_store, embedding_manager)
             smart_retriever_instance = SmartRetriever({
                 'keyword_retriever': keyword_retriever_instance,
                 'basic_retriever': basic_retriever_instance,
                 'collections': vector_store.collections # Pass collection names/objects
             })
             st.success("✅ Retrievers initialized.")
        else:
             st.error("❌ Could not initialize retrievers: Document chunks not available.")
             return None # Cannot proceed without chunks

        # 4. Initialize the main RAG Pipeline with the SmartRetriever
        rag_pipeline_instance = RAGPipeline(smart_retriever_instance)

        st.success("✅ Full RAG Pipeline initialized successfully!")
        return rag_pipeline_instance

    except Exception as e:
        st.error(f"❌ Error initializing RAG Pipeline components: {e}")
        # In case of initialization error, return None or a dummy object
        return None


# ============= STREAMLIT APP =============
def main():
    # Access OpenAI API key securely via Streamlit secrets
    # Ensure OPENAI_API_KEY is set in your Streamlit Cloud secrets (.streamlit/secrets.toml)
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("❌ OpenAI API key not found in Streamlit secrets. Please add it.")
        st.stop() # Stop the app if the key is missing

    # Configure OpenAI client (assuming your RAGPipeline uses the openai library)
    # You might need to pass the key to your RAGPipeline constructor or set an environment variable
    os.environ["OPENAI_API_KEY"] = openai_api_key # Set env var for libraries that read it
    # If your RAGPipeline needs the key passed explicitly, modify its constructor
    # Example: rag_system = load_rag_system(openai_api_key)

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
    # If your RAGPipeline needs the API key, pass it here: load_rag_system(openai_api_key)
    rag_system = load_rag_system()

    if rag_system is None or isinstance(rag_system, DummyClass):
         st.error("Failed to load the RAG system. Please check the initialization code and dependencies.")
         # Still allow basic chat interaction, but responses will be fallbacks
         pass # Continue to render UI

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
            if rag_system and not isinstance(rag_system, DummyClass):
                try:
                    response = rag_system.query(user_input)
                    bot_message = {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources # Ensure sources are included
                    }
                except Exception as e:
                    st.error(f"Error during RAG query: {e}")
                    bot_message = {
                        "role": "assistant",
                        "content": f"An error occurred while processing your request. Details: {e}",
                        "sources": []
                    }
            else:
                 # Fallback if RAG system failed to load or is a dummy
                 bot_message = {
                     "role": "assistant",
                     "content": f"I'm sorry, the research bot is currently unavailable or not fully initialized. Could not process your request about: {user_input}",
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
                        if rag_system and not isinstance(rag_system, DummyClass):
                            try:
                                response = rag_system.query(question)
                                bot_message = {
                                    "role": "assistant",
                                    "content": response.answer,
                                    "sources": response.sources # Ensure sources are included
                                }
                            except Exception as e:
                                st.error(f"Error during RAG query: {e}")
                                bot_message = {
                                    "role": "assistant",
                                    "content": f"An error occurred while processing your request. Details: {e}",
                                    "sources": []
                                }
                        else:
                            # Fallback if RAG system failed to load or is a dummy
                            bot_message = {
                                "role": "assistant",
                                "content": f"I'm sorry, the research bot is currently unavailable or not fully initialized. Could not process your request about: {question}",
                                "sources": []
                            }
                        st.session_state.messages.append(bot_message)
                    st.rerun()


    st.markdown("---")
    st.markdown("🤖 **Research Paper Answer Bot** - Powered by AI | Created by Daniel Ojeda Rosales")

if __name__ == "__main__":
    main()
