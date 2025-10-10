import os
import sys
import streamlit as st
from utils.huggingface_client import client

# Ensure root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# ---------------------------- #
#   ⚙️  CONFIG
# ---------------------------- #
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are CodeGPT — a professional AI coding assistant. "
        "You can write, debug, and explain code in any programming language with clean formatting. "
        "When showing code, always use Markdown fenced code blocks with correct syntax highlighting."
    )
}


# ---------------------------- #
#   🧩 UTILITIES
# ---------------------------- #
def render_message(role, content):
    """Render chat messages with markdown support."""
    with st.chat_message(role):
        st.markdown(content, unsafe_allow_html=True)


def scroll_to_bottom_button():
    """Floating button to scroll to the bottom like ChatGPT."""
    scroll_button_html = """
    <style>
        #scroll-bottom {
            position: fixed;
            bottom: 20px;
            right: 25px;
            background-color: #3b82f6;
            color: white;
            border: none;
            border-radius: 50%;
            padding: 10px 12px;
            cursor: pointer;
            font-size: 20px;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.2);
            transition: 0.2s;
            z-index: 999;
        }
        #scroll-bottom:hover {
            background-color: #2563eb;
        }
    </style>
    <button id="scroll-bottom">⬇️</button>
    <script>
        const btn = window.parent.document.getElementById('scroll-bottom');
        if (btn) btn.onclick = () => window.parent.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    </script>
    """
    st.components.v1.html(scroll_button_html, height=0)


# ---------------------------- #
#   💬 MAIN APP
# ---------------------------- #
def run_code_gpt():
    st.title("🧑‍💻 CodeGPT — AI Coding Assistant")
    st.caption("Powered by **HuggingFace Inference API** | Built with ❤️ using **Streamlit**")

    # Initialize static greeting
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "👋 Hi, I’m **CodeGPT**, your personal AI coding assistant.\n\n"
                    "I can write, debug, and explain code in any programming language. "
                    "Ask me about algorithms, complexity analysis, or best coding practices."
                ),
            }
        ]
        st.session_state.model_used = "HuggingFace Model"

    # Render existing conversation
    for msg in st.session_state.chat_history:
        render_message(msg["role"], msg["content"])

    # User input
    user_input = st.chat_input("💬 Ask me to write or explain code...")
    if user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        render_message("user", user_input)

        try:
            with st.spinner("💡 Generating response..."):
                response = client.chat_completion(
                    messages=[SYSTEM_PROMPT] + st.session_state.chat_history
                )
                reply = response.choices[0].message["content"]
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                render_message("assistant", reply)
        except Exception as e:
            st.error(f"❌ Error generating response: {e}")

    # Floating scroll-to-bottom arrow
    scroll_to_bottom_button()