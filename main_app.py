import streamlit as st
import pickle
import requests
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

# Load RAG system from Task 6
@st.cache_resource
def load_rag_system():
    with open("task6_results.pkl", "rb") as f:
        task6_results = pickle.load(f)
    return task6_results["conversational_rag"]

conv_rag = load_rag_system()

# SerpAPI via requests
def google_search(query: str):
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_key:
        return "⚠️ SerpAPI key not set. Please add it to Streamlit Secrets."

    params = {
        "q": query,
        "api_key": serpapi_key,
        "engine": "google",
        "num": 3
    }

    try:
        response = requests.get("https://serpapi.com/search", params=params)
        result = response.json()

        if "answer_box" in result and "snippet" in result["answer_box"]:
            return result["answer_box"]["snippet"]
        elif "organic_results" in result and len(result["organic_results"]) > 0:
            return "\n\n".join(
                [f"🔗 [{r['title']}]({r['link']})\n> {r.get('snippet', '')}" for r in result["organic_results"]]
            )
        else:
            return "❌ No results found from Google Search."
    except Exception as e:
        return f"❌ Error fetching search results: {e}"

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
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):

                use_web = st.toggle("🌐 Use Google Search instead of local RAG")

                if use_web:
                    answer = google_search(prompt)
                else:
                    answer = conv_rag.query(prompt)

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
