# ===============================================================================
# Research Paper Answer Bot - Simple Deployment Version
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

# ============= MINIMAL RAG COMPONENTS =============
# Since your full RAG system is complex, we'll use a simplified version that works



# ============= STREAMLIT APP =============

def main():
    # Page config
    st.set_page_config(
        page_title="🤖 Research Paper Answer Bot | Daniel Ojeda Rosales",
        page_icon="🧠",
        layout="wide"
    )

    # Custom CSS for dark theme
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #00d4ff;
        text-align: center;
        margin-bottom: 1rem;
    }
    .creator-name {
        font-size: 1.2rem;
        color: #ff6b9d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #667eea;
        color: white;
    }
    .bot-message {
        background-color: #11998e;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">🤖 Research Paper Answer Bot</div>
    <div class="creator-name">✨ Created by Daniel Ojeda Rosales ✨</div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 About")
        st.markdown("""
        This is an AI-powered research assistant that can answer questions about:
        - 🤖 Transformer architectures
        - 🧠 BERT and GPT models
        - 📖 Deep learning research
        - 🔍 Natural language processing
        """)

        st.markdown("### ⚙️ Settings")
        response_style = st.selectbox("Response Style", ["Academic", "Professional", "Casual"])
        show_sources = st.checkbox("Show Sources", value=True)

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Main chat interface
    st.markdown("## 💬 Ask me about AI/ML research papers!")

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Bot:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)

            # Show sources if enabled
            if show_sources and "sources" in message:
                st.markdown("**📚 Sources:**")
                for i, source in enumerate(message["sources"][:3], 1):
                    st.markdown(f"   {i}. {source['filename']} (Page {source['page']}) - Score: {source['score']:.2f}")

    # Chat input
    user_input = st.text_input("💭 Type your question here...", key="user_input")

    col1, col2 = st.columns([1, 4])
    with col1:
        send_button = st.button("🚀 Send", use_container_width=True)

    if send_button and user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get RAG response
        with st.spinner("🤔 Thinking..."):
            rag_system = load_rag_system()
            response = rag_system.query(user_input)

            # Add bot message
            bot_message = {
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources
            }
            st.session_state.messages.append(bot_message)

        st.rerun()

    # Sample questions
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

                    # Get response
                    rag_system = load_rag_system()
                    response = rag_system.query(question)

                    bot_message = {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources
                    }
                    st.session_state.messages.append(bot_message)
                    st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("🤖 **Research Paper Answer Bot** - Powered by AI | Created by Daniel Ojeda Rosales")

if __name__ == "__main__":
    main()
