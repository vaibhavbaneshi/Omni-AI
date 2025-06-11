import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.streamlit_imports import st

from utils.huggingface_client import client

def run_code_gpt():
    st.subheader("🧑‍💻 HuggingFace: Coding Assistant")
    st.info("🔧 Ask any coding-related question, or request code generation in any language.")

    greeting = [
        {"role": "system", "content": "You are a helpful assistant that writes code."},
        {"role": "user", "content": "Greet the user and explain your capabilities."}
    ]

    try:
        with st.spinner("🔍 Initializing assistant..."):
            response = client.chat_completion(messages=greeting)
            assistant_intro = response.choices[0].message['content']
            model_used = response.model

        with st.chat_message("assitant"):
            st.markdown(assistant_intro)
            st.caption(f"Model used: `{model_used}`")

    except Exception as e:
        st.error(f"❌ Failed to initialize assistant: {e}")
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if user_input := st.chat_input("💬 Ask me to write or explain code..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message('user'):
            st.markdown(user_input)

        try:
            with st.spinner("💡 Generating response..."):
                message_thread = [{"role": "system", "content": "You are a helpful assistant that writes code."}]

                message_thread += st.session_state.chat_history

                response = client.chat_completion(messages=message_thread)

                assistant_reply = response.choices[0].message['content']

                st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

            with st.chat_message('assistant'):
                st.markdown(assistant_reply)

        except Exception as e:
            st.error(f"❌ Error generating response: {e}")