import os
import sys
import re
import streamlit as st
import streamlit.components.v1 as components

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.logger import logger
from services.text_extractor import extract_text_from_file, extract_text_from_url
from services.kg_service import kg_service
from utils.scheduler import start_scheduler, ping_neo4j, keep_neo4j_alive

@st.cache_resource
def init_neo4j_driver():
    return keep_neo4j_alive(interval_minutes=30)

driver = init_neo4j_driver()

# Initialize Knowledge Graph service
kg = kg_service()

def run_kg_chatbot():
    st.subheader("🧠 OmniAI - Knowledge Graph Tool")

    # ----------------- Session State Initialization -----------------
    if "input_valid" not in st.session_state:
        st.session_state.input_valid = False
    if "final_chunk" not in st.session_state:
        st.session_state.final_chunk = []
    if "query_kg_flag" not in st.session_state:
        st.session_state.query_kg_flag = False
    if "final_text" not in st.session_state:
        st.session_state.final_text = ""
    if "scheduler_started" not in st.session_state:
        start_scheduler()
        st.session_state.scheduler_started = True
    if "last_input_type" not in st.session_state:
        st.session_state.last_input_type = None

    # ----------------- Input Section -----------------
    input_type = st.radio("Select input type for Knowledge Graph:", ("Text", "URL", "File"))
    user_text = ""
    file_input = None
    url_input = ""

    # Reset graph & flags if input type changes
    if st.session_state.last_input_type != input_type:
        if "graph_html" in st.session_state:
            del st.session_state["graph_html"]
        st.session_state.input_valid = False
        st.session_state.final_chunk = []
        st.session_state.query_kg_flag = False
        st.session_state.last_input_type = input_type

    if input_type == "Text":
        user_text = st.text_area("✍️ Enter text to build KG:")
    elif input_type == "URL":
        url_input = st.text_input("🔗 Enter a URL:")
    elif input_type == "File":
        file_input = st.file_uploader("📂 Upload your KG file", type=["txt", "csv"])

    # ----------------- Input Validation -----------------
    def is_gibberish(text):
        if len(text.strip().split()) < 5:
            return True
        if re.fullmatch(r"[a-zA-Z\s]{0,10}", text.strip()):
            return True
        if len(set(text.strip())) < 5:
            return True
        return False

    if st.button("🔍 Check Neo4j DB Now"):
        ping_neo4j()

    if st.button("✅ Check Input"):
        try:
            if input_type == "Text":
                final_chunk = [user_text]
            elif input_type == "URL":
                final_chunk = extract_text_from_url(url_input) if url_input else []
            elif input_type == "File":
                final_chunk = extract_text_from_file(file_input) if file_input else []

            if "graph_html" in st.session_state:
                del st.session_state["graph_html"]

            full_text = " ".join(final_chunk)
            if not full_text or is_gibberish(full_text):
                st.session_state.input_valid = False
                st.warning("⚠️ The input is incomplete or unclear. Please provide more meaningful information.")
            else:
                kg.reset_kg()
                st.session_state.input_valid = True
                st.session_state.final_chunk = final_chunk
                st.session_state.query_kg_flag = False
                st.success("✅ Input is valid and ready to build the KG!")
        except Exception as e:
            st.error(f"❌ Error processing input: {e}")
            st.session_state.input_valid = False

    # ----------------- Build KG -----------------
    if st.session_state.input_valid:
        if st.button("🚀 Build Knowledge Graph"):
            try:
                for chunk in st.session_state.final_chunk:
                    kg.build_kg(chunk)
                st.session_state.query_kg_flag = True
            except Exception as e:
                st.error(f"❌ KG build failed: {e}")
    else:
        st.button("🚀 Build Knowledge Graph", disabled=True)
        st.info("ℹ️ Please check the input first.")

    # ----------------- Visualize KG -----------------
    if "graph_html" in st.session_state:
        components.html(st.session_state["graph_html"], height=600, width=1000, scrolling=True)

    # ----------------- Show Neo4j Schema (only after KG build) -----------------
    if st.session_state.query_kg_flag:
        with st.expander("🗂️ View Current Neo4j Schema"):
            try:
                schema_snippet = kg.generate_schema_snippet()
                st.text_area(
                    "Current Labels, Properties, and Relationships",
                    value=schema_snippet,
                    height=400
                )
                if st.button("🔄 Refresh Schema"):
                    schema_snippet = kg.generate_schema_snippet()
                    st.success("✅ Schema refreshed!")
            except Exception as e:
                logger.error(f"Failed to fetch schema: {e}")
                st.error("⚠️ Could not fetch Neo4j schema.")

    # ----------------- Query KG -----------------
    if st.session_state.query_kg_flag:
        user_question = st.text_input("🔍 Ask a question about the Knowledge Graph:")
        if st.button("Query Graph"):
            if not user_question or user_question.strip() == "":
                st.warning("⚠️ Please enter a question.")
            else:
                try:
                    result = kg.generate_query(user_question)
                    st.write(result)
                except Exception as e:
                    logger.error(f"Query failed: {e}")
                    st.error("❌ Error querying Knowledge Graph. Check logs.")