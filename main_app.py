import streamlit as st
import pickle
import requests  # ✅ Fallback for SerpAPI

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

# ✅ Load RAG system from Task 6
@st.cache_resource
def load_rag_system():
    with open("task6_results.pkl", "rb") as f:
        task6_results = pickle.load(f)
    return task6_results["conversational_rag"]

conv_rag = load_rag_system()

# ✅ Google Search using requests
def google_search(query: str, serpapi_key: str, num_results: int = 3):
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": serpapi_key,
        "engine": "google",
        "num": num_results
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data.get("organic_results", [])

# Session init
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Header
st.markdown('<h1 class="main-header">🧠 AI Research Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="author-header">👨‍💻 Developed by Daniel Ojeda Rosales</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🤖 Chat", "📚 Documents", "📊 Analytics"])

with tab1:
    st.markdown("### 💬 Chat with AI")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):

                # API Key inputs
                if "serpapi_key" not in st.session_state:
                    st.session_state.serpapi_key = st.text_input("🔑 Enter your SerpAPI Key", type="password")
                    st.stop()

                use_web = st.toggle("🌐 Use Google Search instead of local RAG")

                if use_web and st.session_state.serpapi_key:
                    with st.spinner("🌐 Searching Google..."):
                        results = google_search(prompt, st.session_state.serpapi_key)
                        if not results:
                            answer = "❌ No results found from Google Search."
                        else:
                            answer = "\n\n".join(
                                [f"🔗 [{r['title']}]({r['link']})\n> {r.get('snippet', '')}" for r in results]
                            )
                else:
                    with st.spinner("🔎 Querying RAG system..."):
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
