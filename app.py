import streamlit as st
from common.core_imports import run_pdf_rag, run_search_engine, run_math_gpt, run_code_gpt, run_docchat_with_filters, run_kg_chatbot, run_fine_tune, run_youtube_blog

st.set_page_config(page_title="Omni-AI", layout="wide")
st.title("Omni-AI")
st.subheader("All Your AI Needs. One OmniBrain.")

st.sidebar.title("Choose a Feature")

# === TOOL REGISTRY ===
TOOL_REGISTRY = {
    "✅ Chatbots": {
        "✅ PDF Q&A Chatbot": lambda: run_pdf_rag,
        "✅ Multi-Agent RAG Chatbot": lambda: run_search_engine,
        "✅ Multilingual Code Assistant": lambda: run_code_gpt
    },
    "📄 Document & Web Intelligence": {
        "📄 DocChat with Metadata Filters": lambda: run_docchat_with_filters
    },
    "🔢 Math & Reasoning": {
        "🔢 Integrated Wikipedia + Math + Reasoning Agent": lambda: run_math_gpt
    },
    "🧠 Knowledge & Graph-based Tools": {
        "🧠 Knowledge Graph Build and ChatBot": lambda: run_kg_chatbot
    },
    "🧪 Fine-Tuning Playground": {
        "🧪 Fine-tune LLM with Custom Data": lambda: run_fine_tune
    },
    "🧠 Multi-Agent Systems": {
        "🧠 YouTube-to-Blog Writer": run_youtube_blog
    }
}

# Step 1: Select Category
feature = st.sidebar.selectbox("🔍 Select a GenAI Tool", ["--Select--"] + list(TOOL_REGISTRY.keys()))

genai_tool = "--Select--"
if feature != "--Select--":
    # Step 2: Select Tool from selected category
    tool_names = ["--Select--"] + list(TOOL_REGISTRY[feature].keys())
    genai_tool = st.sidebar.selectbox("Choose a Tool", tool_names)

# Step 3: Trigger tool if selected
if feature != "--Select--" and genai_tool != "--Select--":
    with st.spinner(f"🔄 Loading {genai_tool}... Please wait..."):
        TOOL_REGISTRY[feature][genai_tool]()
