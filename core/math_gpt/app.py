import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.langchain_imports import (
    PromptTemplate, AgentType, Tool, initialize_agent, StreamlitCallbackHandler, LLMChain
)
from utils.llm import llm
from utils.tools import wikipedia_tool, calculator


def run_math_gpt():
    st.set_page_config(page_title="Math & Reasoning Assistant", page_icon="🧮", layout="wide")
    st.title("🧮 Text to Math/Reasoning Assistant")
    st.caption("Solve math, logic, and reasoning problems step-by-step with explanations.")

    # --- Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 Hi, I’m your Math/Logical Reasoning Assistant! Ask me anything."
        }]
        st.session_state.model_name = "llama-3.1-8b-instant"

    # --- Model Check ---
    try:
        result = llm.invoke("Hello!")
        model_name = getattr(result, "model", None)
        if hasattr(result, "response_metadata"):
            model_name = model_name or result.response_metadata.get("model_name")
        st.session_state.model_name = model_name or "Unknown"
        st.success("✅ Model connected successfully!")
        st.info(f"🧠 Model Used: {st.session_state.model_name}")
    except Exception as e:
        st.warning("⚠️ Couldn't detect model name.")
        st.exception(e)

    # --- Agent Creation ---
    reasoning_prompt = PromptTemplate(
        input_variables=['question'],
        template="""
        You are an intelligent assistant tasked with solving the user's mathematical or logical reasoning question.
        Think step-by-step, provide a detailed explanation, and display the solution clearly.

        Question: {question}
        Answer:
        """
    )
    reasoning_chain = LLMChain(llm=llm, prompt=reasoning_prompt)
    reasoning_tool = Tool(
        name="Reasoning Tool",
        func=reasoning_chain.run,
        description="Solves logic and reasoning problems with detailed step-by-step answers."
    )

    assistant_agent = initialize_agent(
        tools=[wikipedia_tool, calculator, reasoning_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True
    )

    # --- Render Chat History ---
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    # --- Chat Input ---
    if user_input := st.chat_input("💬 Type your math or reasoning question... (Ask question like mathematical theorems or reasoning questions)"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # --- Agent Response ---
        try:
            with st.spinner("🧠 Thinking..."):
                callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = assistant_agent.run(user_input, callbacks=[callback])
                st.session_state.messages.append({"role": "assistant", "content": response})

                with st.chat_message("assistant"):
                    st.success(f"✅ Result: {response}")
        except Exception as e:
            st.error("❌ Error generating response.")
            st.exception(e)

    # --- Scroll-to-Bottom Button ---
    scroll_html = """
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
    st.components.v1.html(scroll_html, height=0)