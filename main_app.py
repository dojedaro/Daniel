import streamlit as st
from serpapi import GoogleSearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

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

# Load RAG system directly (rebuild instead of pickle)
@st.cache_resource
def build_rag_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=st.session_state.openai_api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

# SerpAPI search function
def google_search(query: str, serpapi_key: str, num_results: int = 3):
    params = {
        "q": query,
        "api_key": serpapi_key,
        "engine": "google",
        "num": num_results
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("organic_results", [])

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Header
st.markdown('<h1 class="main-header">🧠 AI Research Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="author-header">👨‍💻 Developed by Daniel Ojeda Rosales</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🤖 Chat", "📚 Documents", "📊 Analytics"])

with tab1:
    st.markdown("### 💬 Chat with AI")

    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                # Ask for API keys
                if "openai_api_key" not in st.session_state:
                    st.session_state.openai_api_key = st.text_input("🔐 Enter your OpenAI API Key", type="password")
                    st.stop()
                if "serpapi_key" not in st.session_state:
                    st.session_state.serpapi_key = st.text_input("🔑 Enter your SerpAPI Key", type="password")
                    st.stop()

                use_web = st.toggle("🌐 Use Google Search instead of local RAG")

                if use_web:
                    results = google_search(prompt, st.session_state.serpapi_key)
                    if not results:
                        answer = "❌ No Google results found."
                    else:
                        answer = "

".join(
                            [f"🔗 [{r['title']}]({r['link']})
> {r.get('snippet', '')}" for r in results]
                        )
                else:
                    if "qa_chain" not in st.session_state:
                        st.session_state.qa_chain = build_rag_system()
                    result = st.session_state.qa_chain(prompt)
                    answer = result["result"]

                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

with tab2:
    st.markdown("### 📚 Document Upload")
    uploaded_files = st.file_uploader("Upload PDFs or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} file(s). (Parsing not implemented here)")

with tab3:
    st.markdown("### 📊 Analytics")
    if st.session_state.messages:
        st.metric("Total Messages", len(st.session_state.messages))
    else:
        st.info("No messages yet.")
