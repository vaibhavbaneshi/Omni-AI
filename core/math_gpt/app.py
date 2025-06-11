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

    st.subheader("Text To Math Problem/Reasoning Assistant Solver Agent Using Google Gemma 2 integrated with Wikipedia Tool")

    try:
        result = llm.invoke("Say Hello!")
        st.success(result.content)
        st.success("Model Used: " + result.response_metadata['model_name'])

    except Exception as e:
        st.exception(f"Exception {e}")


    prompt = """

                Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation and display it point wise for the question below
                Question:{question}
                Answer:

             """
    
    prompt_template = PromptTemplate(
        input_variables=['question'],
        template=prompt
    )

    chain=LLMChain(llm=llm, prompt=prompt_template)

    reasoning_tool=Tool(
        name='Reasoning Tool',
        func=chain.run,
        description='A tool for answering logic-based and reasoning questions.'
    )

    assistant_agent=initialize_agent(
        tools=[wikipedia_tool, calculator, reasoning_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_error=True
    )

    if "messages" not in st.session_state:
        st.session_state['messages']=[
            {
                "role": "assistant",
                "content": "I'm a Math/Logical Reasoning ChatBot who can answer all your Math/Logical Reasoning Questions."
            }
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])

    if prompt:=st.chat_input(placeholder="I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?"
    ):
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role:": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response=assistant_agent(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({"role:": "assistant", "content": response})

            st.write('Response: ')
            st.success('Result: ' + response['output'])