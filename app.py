import streamlit as st

# =======================
# PAGE CONFIG
# =======================
if "page_config_set" not in st.session_state:
    st.set_page_config(
        page_title="Omni-AI",
        page_icon="🤖",
        layout="wide"
    )
    st.session_state.page_config_set = True
    
from common.core_imports import (
    run_pdf_rag,
    run_search_engine,
    run_math_gpt,
    run_code_gpt,
    run_docchat_with_filters,
    run_kg_chatbot,
    run_fine_tune,
    run_youtube_blog
)

import threading
import time
from neo4j import GraphDatabase
from configs.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# ===============================
# ⚙️ GLOBAL NEO4J CONNECTION HANDLER
# ===============================
driver = None
driver_lock = threading.Lock()

def get_driver():
    """Ensure the Neo4j driver is always active and reconnects if needed."""
    global driver
    with driver_lock:
        if driver is None:
            try:
                driver = GraphDatabase.driver(
                    NEO4J_URI,
                    auth=(NEO4J_USER, NEO4J_PASSWORD),
                    max_connection_lifetime=3600,
                    connection_timeout=15
                )
                print("🟢 Neo4j driver initialized successfully.")
            except Exception as e:
                print(f"🔴 Failed to initialize Neo4j driver: {e}")
                driver = None
        return driver

def ping_neo4j():
    """Continuously pings the Neo4j DB every 30 minutes to prevent idle timeout."""
    while True:
        try:
            drv = get_driver()
            if drv:
                with drv.session() as session:
                    session.run("RETURN 1 AS keep_alive")
                    print(f"🟢 [{time.strftime('%Y-%m-%d %H:%M:%S')}] Neo4j keep-alive ping sent.")
            else:
                print("🔴 No active Neo4j driver — retrying next cycle.")
        except Exception as e:
            print(f"⚠️ Neo4j keep-alive failed: {e}")
            with driver_lock:
                global driver
                driver = None
        time.sleep(1800)  # Ping every 30 minutes

# Start background thread
threading.Thread(target=ping_neo4j, daemon=True).start()

# =======================
# CSS STYLING
# =======================
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        color: #00AEEF;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #ccc;
        margin-bottom: 30px;
    }
    .stCard {
        background-color: #0E1117;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0px 4px 16px rgba(0,0,0,0.25);
        transition: all 0.25s ease-in-out;
        height: 100%;
    }
    .stCard:hover {
        transform: translateY(-5px);
        border-color: #00AEEF;
        box-shadow: 0 0 25px rgba(0,174,239,0.2);
    }
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    .card-desc {
        color: #ccc;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    </style>
""", unsafe_allow_html=True)

# =======================
# HEADER
# =======================
st.markdown('<h1 class="main-title">🤖 Omni-AI</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">All Your AI Needs — One OmniBrain 🧠</div>', unsafe_allow_html=True)
st.divider()

# =====================
# ⚙️ TOOL REGISTRY (lazy-loaded)
# =====================
@st.cache_resource(show_spinner=False)
def load_tools():
    return {
        "✅ Chatbots": {
            "📘 PDF Q&A Chatbot": run_pdf_rag,
            "🌐 Multi-Agent RAG Chatbot": run_search_engine,
            "💻 Multilingual Code Assistant": run_code_gpt
        },
        "📄 Document & Web Intelligence": {
            "📑 DocChat with Metadata Filters": run_docchat_with_filters
        },
        "🔢 Math & Reasoning": {
            "🧮 Integrated Wikipedia + Math Agent": run_math_gpt
        },
        "🧠 Knowledge & Graph-based Tools": {
            "🔗 Knowledge Graph Build and ChatBot": run_kg_chatbot
        },
        "🎥 Content & Creativity": {
            "📝 YouTube-to-Blog Writer": run_youtube_blog
        }
    }

TOOL_REGISTRY = load_tools()

# =====================
# 🧭 SIDEBAR SELECTION
# =====================
st.sidebar.title("🔍 Choose a Feature")

feature = st.sidebar.selectbox("Select a Category", ["--Select--"] + list(TOOL_REGISTRY.keys()))

genai_tool = "--Select--"
if feature != "--Select--":
    genai_tool = st.sidebar.selectbox("Select a Tool", ["--Select--"] + list(TOOL_REGISTRY[feature].keys()))

# =====================
# 🚀 LOAD TOOL DYNAMICALLY
# =====================
if feature != "--Select--" and genai_tool != "--Select--":
    #st.markdown(f"<div class='loader'>🌀 Launching {genai_tool}...</div>", unsafe_allow_html=True)
    TOOL_REGISTRY[feature][genai_tool]()

# =======================
# LANDING PAGE (if nothing selected)
# =======================
if feature == "--Select--":
    st.markdown("### ✨ Explore Omni-AI Tools")

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown('<div class="stCard"><div class="card-title">🧩 Chatbots</div><div class="card-desc">Intelligent assistants that help you interact with PDFs, generate code, and explore multi-agent reasoning.</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="stCard"><div class="card-title">📚 Document & Web Intelligence</div><div class="card-desc">Upload, filter, and chat with complex documents or web data effortlessly.</div></div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="stCard"><div class="card-title">🧠 Knowledge & Graph Tools</div><div class="card-desc">Create knowledge graphs, query relationships, and visualize knowledge structures dynamically.</div></div>', unsafe_allow_html=True)

    st.write("")
    col4, col5, col6 = st.columns(3, gap="large")

    with col4:
        st.markdown('<div class="stCard"><div class="card-title">🔢 Math & Reasoning</div><div class="card-desc">Perform advanced reasoning with integrated Wikipedia and mathematical capabilities.</div></div>', unsafe_allow_html=True)

    with col5:
        st.markdown('<div class="stCard"><div class="card-title">🧪 Fine-Tuning Playground</div><div class="card-desc">Train, evaluate, and experiment with fine-tuning your own models easily.</div></div>', unsafe_allow_html=True)

    with col6:
        st.markdown('<div class="stCard"><div class="card-title">🎥 Creative Systems</div><div class="card-desc">Transform YouTube videos into blogs or scripts using open-source caption parsing and summarization.</div></div>', unsafe_allow_html=True)
