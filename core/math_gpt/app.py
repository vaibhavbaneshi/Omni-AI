import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.streamlit_imports import st
from common.langchain_imports import (
    PromptTemplate, AgentType, Tool, initialize_agent, StreamlitCallbackHandler, LLMChain
)
from utils.llm import llm
from utils.tools import wikipedia_tool, calculator

def run_math_gpt():
    st.subheader("🧮 Text to Math Problem / Reasoning Assistant (Gemma 2 + Wikipedia)")

    # --- Model check ---
    try:
        result = llm.invoke("Say Hello!")
        # Handle both dict and simple text outputs
        model_name = None

        # Try multiple safe checks
        if hasattr(result, "response_metadata") and isinstance(result.response_metadata, dict):
            model_name = result.response_metadata.get("model") or result.response_metadata.get("model_name")
        elif hasattr(result, "model"):
            model_name = result.model

        st.success(result.content if hasattr(result, "content") else str(result))
        st.info(f"🧠 Model Used: {model_name or 'Unknown'}")

    except Exception as e:
        st.warning("⚠️ Couldn't detect model name.")
        st.exception(e)


    # --- Prompt Template ---
    prompt = """
    You are an intelligent agent tasked with solving the user's mathematical or logical reasoning question.
    Think step-by-step, provide a detailed explanation, and display the solution clearly and neatly.

    Question: {question}
    Answer:
    """

    prompt_template = PromptTemplate(
        input_variables=['question'],
        template=prompt
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)

    reasoning_tool = Tool(
        name='Reasoning Tool',
        func=chain.run,
        description='A tool for answering logic-based and reasoning questions.'
    )

    assistant_agent = initialize_agent(
        tools=[wikipedia_tool, calculator, reasoning_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True
    )

    # --- Session State ---
    if "messages" not in st.session_state:
        st.session_state['messages'] = [{
            "role": "assistant",
            "content": "I'm a Math/Logical Reasoning ChatBot ready to solve your questions!"
        }]

    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])

    # --- Chat Input ---
    if prompt := st.chat_input("Ask me any math or reasoning question..."):
        with st.spinner("🧠 Thinking..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(prompt, callbacks=[st_cb])

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").success(f"✅ Result: {response}")