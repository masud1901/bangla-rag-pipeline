import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Multilingual RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- API Configuration ---
FASTAPI_BASE_URL = "http://rag-api:8000"
QUERY_URL = f"{FASTAPI_BASE_URL}/query"
HEALTH_URL = f"{FASTAPI_BASE_URL}/health"

# --- Helper Functions ---
def get_system_health():
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def detect_language(text):
    bengali_chars = set('অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৎংঃ')
    bengali_count = sum(1 for char in text if char in bengali_chars)
    return "bn" if bengali_count > len(text) * 0.1 else "en"

def get_answer(query: str, language: str = None, include_sources: bool = True):
    if language is None:
        language = detect_language(query)
    payload = {
        "query": query,
        "language": language,
        "include_sources": include_sources,
        "top_k": 5
    }
    try:
        start_time = time.time()
        response = requests.post(QUERY_URL, json=payload, timeout=120)
        end_time = time.time()
        if response.status_code == 200:
            result = response.json()
            result["client_processing_time"] = (end_time - start_time) * 1000
            return result
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
        return None

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "current_language" not in st.session_state:
    st.session_state.current_language = "auto"

# --- Sidebar ---
with st.sidebar:
    st.title("📚 Multilingual RAG")
    st.markdown("Ask questions in **বাংলা** or **English** about the textbook.")
    
    # System Status
    st.subheader("🔧 System Status")
    health_data = get_system_health()
    if health_data:
        status = health_data.get("status", "unknown")
        if status == "healthy":
            st.success("✅ System Online")
        else:
            st.error("❌ System Offline")
        
        # Show vector store stats
        config = health_data.get("config", {})
        if "pinecone_index" in config:
            st.info(f"📊 Index: {config['pinecone_index']}")
    else:
        st.error("❌ Cannot connect to API")
    
    # Language Selection
    st.subheader("🌐 Language")
    language_option = st.selectbox(
        "Response Language",
        ["auto", "en", "bn"],
        format_func=lambda x: {
            "auto": "🔄 Auto-detect",
            "en": "🇺🇸 English", 
            "bn": "🇧🇩 বাংলা"
        }[x]
    )
    st.session_state.current_language = language_option
    
    # Sample Questions
    st.subheader("💡 Sample Questions")
    sample_questions = {
        "🇧🇩 Bengali": [
            "এই বই কী সম্পর্কে?",
            "বাংলা সাহিত্যের ইতিহাস কী?",
            "শিক্ষাক্রমের উদ্দেশ্য কী?",
        ],
        "🇺🇸 English": [
            "What is this book about?",
            "What are the main topics?",
            "Who are the authors?",
        ]
    }
    
    for lang, questions in sample_questions.items():
        st.markdown(f"**{lang}:**")
        for q in questions:
            if st.button(q, key=f"sample_{q}", use_container_width=True):
                st.session_state.sample_query = q
                st.rerun()
    
    # Stats
    if st.session_state.query_history:
        st.subheader("📈 Statistics")
        df = pd.DataFrame(st.session_state.query_history)
        avg_time = df['processing_time'].mean()
        st.metric("Avg Response", f"{avg_time:.0f}ms")
        st.metric("Total Queries", len(df))
    
    # Actions
    st.subheader("⚙️ Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("📊 Export", use_container_width=True):
            if st.session_state.messages:
                chat_data = json.dumps(st.session_state.messages, default=str, indent=2)
                st.download_button(
                    label="💾 Download",
                    data=chat_data,
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

# --- Main Chat Interface ---
st.title("💬 Multilingual RAG Chat")
st.caption("Ask questions about HSC Bangla Literature in বাংলা or English")

# Handle sample question selection
if 'sample_query' in st.session_state:
    prompt = st.session_state.sample_query
    del st.session_state.sample_query
else:
    prompt = None

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander(f"📚 Sources ({len(message['sources'])} found)"):
                for i, chunk in enumerate(message["sources"], 1):
                    st.markdown(f"**Page {chunk.get('page_number', 'N/A')}** (Score: {chunk.get('relevance_score', 0):.2f})")
                    st.markdown(f"> {chunk.get('text', 'No text available')[:200]}...")
        
        # Show metadata if available
        if message["role"] == "assistant" and "metadata" in message:
            with st.expander("🔍 Details"):
                meta = message["metadata"]
                st.json(meta)

# Chat input
if not prompt:
    prompt = st.chat_input("Ask your question in English or বাংলায় প্রশ্ন করুন...")

if prompt:
    # Add user message
    detected_lang = detect_language(prompt)
    use_lang = detected_lang if st.session_state.current_language == "auto" else st.session_state.current_language
    
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "language": use_lang,
        "timestamp": datetime.now()
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            api_response = get_answer(prompt, use_lang, include_sources=True)
        
        if api_response:
            answer = api_response.get("answer", "No answer found.")
            sources = api_response.get("source_chunks", [])
            
            st.markdown(answer)
            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources,
                "metadata": {
                    "processing_time_ms": api_response.get("processing_time_ms", 0),
                    "retrieved_chunks_count": len(sources),
                    "embedding_model": api_response.get("model_info", {}).get("embedding_model", "N/A"),
                    "llm_model": api_response.get("model_info", {}).get("llm", "N/A")
                },
                "timestamp": datetime.now()
            })
            
            # Update query history
            st.session_state.query_history.append({
                "query": prompt,
                "language": use_lang,
                "processing_time": api_response.get("processing_time_ms", 0),
                "chunks_found": len(sources),
                "timestamp": datetime.now()
            })
            
            # Show sources if available
            if sources:
                with st.expander(f"📚 Sources ({len(sources)} found)"):
                    for i, chunk in enumerate(sources, 1):
                        st.markdown(f"**Page {chunk.get('page_number', 'N/A')}** (Score: {chunk.get('relevance_score', 0):.2f})")
                        st.markdown(f"> {chunk.get('text', 'No text available')[:200]}...")
            
            # Show metadata
            with st.expander("🔍 Response Details"):
                st.json({
                    "Processing Time (ms)": api_response.get("processing_time_ms", 0),
                    "Chunks Retrieved": len(sources),
                    "Model": api_response.get("model_info", {}).get("embedding_model", "N/A")
                })
        else:
            error_msg = "❌ Sorry, I encountered an error. Please check the system status and try again."
            st.markdown(error_msg)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": error_msg,
                "timestamp": datetime.now()
            })

# Footer
st.markdown("---")
st.markdown("**Made with ❤️ using Streamlit & FastAPI**") 