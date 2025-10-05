import streamlit as st
from common.core_imports import run_pdf_rag, run_search_engine, run_math_gpt, run_code_gpt, run_docchat_with_filters, run_kg_chatbot, run_fine_tune

st.set_page_config(page_title="Omni-AI", layout="wide")
st.title("Omni-AI")
st.title("All Your AI Needs. One OmniBrain.")

st.sidebar.title("Choose a Feature")

# === TOOL REGISTRY ===
TOOL_REGISTRY = {
    "✅ Chatbots": {
        "✅ PDF Q&A Chatbot": run_pdf_rag,
        "✅ Multi-Agent RAG Chatbot": run_search_engine,
        "✅ Multilingual Code Assistant": run_code_gpt
    },
    "📄 Document & Web Intelligence": {
        "📄 DocChat with Metadata Filters": run_docchat_with_filters
    },
    "🔢 Math & Reasoning": {
        "🔢 Integrated Wikipedia + Math + Reasoning Agent": run_math_gpt
    },
    "🧠 Knowledge & Graph-based Tools": {
        "🧠 Knowledge Graph Build and ChatBot": run_kg_chatbot,
    },
    "🧪 Fine-Tuning Playground": {
        "🧪 Fine-tune LLM with Custom Data": run_fine_tune
    },
    "🧠 Multi-Agent Systems": {
        "🧠 YouTube-to-Blog Writer": st.write("YT-Blog Writer coming soon..."),
        "🧠 Research Assistant with Search + Summary Agents": st.write("Research Assistant coming soon..."),
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
