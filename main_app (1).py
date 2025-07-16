import streamlit as st
from serpapi import GoogleSearch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile

# Page config
st.set_page_config(
    page_title="🧠 AI Research Assistant | Daniel Ojeda Rosales",
    page_icon="🤖",
    layout="wide"
)

# Styling
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .main-header {
        font-size: 3rem; font-weight: bold;
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

# Function to build RAG pipeline from uploaded PDFs
def build_rag_from_uploads(uploaded_files):
    all_docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()
            all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embeddings)
    vectorstore.add_documents(split_docs)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=st.session_state.openai_api_key)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Google Search fallback
def google_search(query, serpapi_key):
    search = GoogleSearch({
        "q": query,
        "api_key": serpapi_key,
        "engine": "google",
        "num": 3
    })
    return search.get_dict().get("organic_results", [])

# Initialize session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Header
st.markdown('<h1 class="main-header">🧠 AI Research Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="author-header">👨‍💻 Developed by Daniel Ojeda Rosales</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🤖 Chat", "📚 Documents"])

with tab2:
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        st.session_state.uploaded_docs = uploaded_files
        st.success(f"{len(uploaded_files)} file(s) uploaded!")

with tab1:
    if prompt := st.chat_input("Ask your research question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):

                if "openai_api_key" not in st.session_state:
                    st.session_state.openai_api_key = st.text_input("🔐 Enter OpenAI API Key", type="password")
                    st.stop()
                if "serpapi_key" not in st.session_state:
                    st.session_state.serpapi_key = st.text_input("🔑 Enter SerpAPI Key", type="password")
                    st.stop()

                use_google = st.toggle("🌐 Use Google Search")
                answer = "⚠️ No documents uploaded."

                if use_google:
                    results = google_search(prompt, st.session_state.serpapi_key)
                    if not results:
                        answer = "❌ No results found from Google."
                    else:
                        answer = "

".join([f"🔗 [{r['title']}]({r['link']})
> {r.get('snippet','')}" for r in results])
                elif "uploaded_docs" in st.session_state:
                    qa_chain = build_rag_from_uploads(st.session_state.uploaded_docs)
                    result = qa_chain(prompt)
                    answer = result["result"]

                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
