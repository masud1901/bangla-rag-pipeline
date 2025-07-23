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
DEBUG_URL = f"{FASTAPI_BASE_URL}/debug/retrieve"

# --- Helper Functions ---
@st.cache_data(ttl=10) # Cache for 10 seconds
def get_system_health():
    """Fetches and caches system health data."""
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        return None
    return None

def get_answer(query: str, language: str = "auto"):
    """Posts a query to the backend and returns the response."""
    payload = { "query": query, "language": language }
    try:
        response = requests.post(QUERY_URL, json=payload, timeout=120)
        return response.json() if response.status_code == 200 else {"error": response.text, "status_code": response.status_code}
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "status_code": 503}

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# --- Sidebar ---
with st.sidebar:
    st.title("📚 Multilingual RAG")
    st.markdown("Ask questions in **বাংলা** or **English** about the textbook.")
    
    # System Status
    st.subheader("🔧 System Status")
    health_data = get_system_health()
    if health_data and health_data.get("status") == "healthy":
        st.success("✅ System Online")
        
        # Display Vector Store Stats
        stats = health_data.get("config", {}).get("index_stats", {})
        if stats:
            st.markdown("**Vector Store:**")
            col1, col2 = st.columns(2)
            col1.metric("Cohere Vectors", stats.get("cohere_vector_count", 0))
            col2.metric("OpenAI Vectors", stats.get("openai_vector_count", 0))
            st.metric("Total Vectors", stats.get("total_vectors", 0))
    else:
        st.error("❌ System Offline or Degraded")

    # Sample Questions
    st.subheader("💡 Sample Questions")
    sample_questions_bn = [
        "মানবিক চেতনা সম্পর্কে কী বলা হয়েছে?",
        "মানব-কল্যাণের অর্থ কী?",
        "সম্পর্কের গুরুত্ব কী?",
    ]
    sample_questions_en = [
        "What is human consciousness?",
        "What is human welfare?",
        "What is the importance of relationships?",
    ]
    
    st.markdown("**🇧🇩 Bengali:**")
    for q in sample_questions_bn:
        if st.button(q, key=q, use_container_width=True):
            st.session_state.current_query = q
            st.rerun()

    st.markdown("**🇺🇸 English:**")
    for q in sample_questions_en:
        if st.button(q, key=q, use_container_width=True):
            st.session_state.current_query = q
            st.rerun()

    # Actions
    st.subheader("⚙️ Actions")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_history = []
        st.rerun()

# --- Main Chat Interface ---
st.title("💬 Multilingual RAG Chat")
st.caption("Ask questions about HSC Bangla Literature in বাংলা or English")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new query
if "current_query" in st.session_state:
    prompt = st.session_state.current_query
    del st.session_state.current_query
else:
    prompt = st.chat_input("Ask your question in English or বাংলায় প্রশ্ন করুন...")

if prompt:
    # Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            response = get_answer(prompt)
            
            if "error" in response:
                st.error(f"API Error: {response.get('status_code')} - {response.get('error')}")
                answer = "Sorry, I encountered an error. Please check the system status."
            else:
                answer = response.get("answer", "I couldn't find an answer.")
                st.markdown(answer)

                # Display sources
                if response.get("source_chunks"):
                    with st.expander(f"📚 Sources ({len(response['source_chunks'])} found)"):
                        for chunk in response["source_chunks"]:
                            st.info(f"**Page {chunk.get('page_number', 'N/A')}** (Score: {chunk.get('relevance_score', 0):.2f})")
                            st.markdown(f"> {chunk.get('text', '')}")

                # Display debug info
                with st.expander("🔍 Debug Info"):
                    st.json({
                        "Processing Time (ms)": response.get("processing_time_ms"),
                        "Model Info": response.get("model_info"),
                        "Retrieval Stats": response.get("retrieval_stats")
                    })

            # Add assistant message to state
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Update query history for sidebar stats
            st.session_state.query_history.append({
                "query": prompt,
                "processing_time": response.get("processing_time_ms", 0)
            })
            
            # Rerun to update sidebar stats
            st.rerun() 