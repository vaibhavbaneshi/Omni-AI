import streamlit as st
from document_qa_rag.app import run_pdf_rag
st.set_page_config(page_title="Omni-AI", layout="wide")
st.title("Omni-AI")
st.title("All Your AI Needs. One OmniBrain.")

st.sidebar.title("Choose a Feature")
genai_tool = ""

# === TOOL REGISTRY ===
TOOL_REGISTRY = {
    "✅ Chatbots": {
        "✅ Conversational Chatbot": lambda: st.write("Conversational Chatbot coming soon..."),
        "✅ PDF Q&A Chatbot": run_pdf_rag,
        "✅ Multi-Agent RAG Chatbot": lambda: st.write("Multi-Agent RAG coming soon..."),
        "✅ Multilingual Code Assistant": lambda: st.write("Code Assistant coming soon..."),
        "✅ Multi-LLM Chatbot": lambda: st.write("Multi-LLM Chatbot coming soon...")
    },
    "📄 Document & Web Intelligence": {
        "📄 Website Summarizer": lambda: st.write("Website Summarizer coming soon..."),
        "📄 YouTube Video Summarizer": lambda: st.write("YouTube Summarizer coming soon..."),
        "📄 Text Summarizer": lambda: st.write("Text Summarizer coming soon..."),
        "📄 Document Uploader & Multi-PDF Search": lambda: st.write("Multi-PDF Search coming soon..."),
        "📄 DocChat with Metadata Filters": lambda: st.write("Metadata DocChat coming soon...")
    },
    "🔢 Math & Reasoning": {
        "🔢 Math Problem Solver": lambda: st.write("Math Solver coming soon..."),
        "🔢 YouTube Video Summarizer": lambda: st.write("YouTube Video Summarizer coming soon..."),
        "🔢 Logical Reasoning Assistant": lambda: st.write("Reasoning Assistant coming soon..."),
        "🔢 Integrated Wikipedia + Math + Reasoning Agent": lambda: st.write("Wiki+Math Agent coming soon...")
    },
    "🧠 Knowledge & Graph-based Tools": {
        "🧠 Knowledge Graph Builder": lambda: st.write("KG Builder coming soon..."),
        "🧠 Graph Query Chatbot": lambda: st.write("Graph Query Bot coming soon..."),
        "🧠 Graph-based Multi-hop Reasoner": lambda: st.write("Multi-hop Reasoner coming soon...")
    },
    "🧪 Fine-Tuning Playground": {
        "🧪 Fine-tune LLM with Custom Data": lambda: st.write("Fine-tuning coming soon..."),
        "🧪 Prompt Debugger with LangSmith": lambda: st.write("LangSmith Debugger coming soon..."),
        "🧪 Quantization Visualizer": lambda: st.write("Quantization Visualizer coming soon...")
    },
    "🧠 Multi-Agent Systems": {
        "🧠 YouTube-to-Blog Writer": lambda: st.write("YT-Blog Writer coming soon..."),
        "🧠 Research Assistant with Search + Summary Agents": lambda: st.write("Research Assistant coming soon..."),
        "🧠 Multi-Agent File Analyzer": lambda: st.write("File Analyzer coming soon...")
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
