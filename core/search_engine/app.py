def run_search_engine():
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    from common.streamlit_imports import st
    from common.langchain_imports import ChatGroq, ArxivAPIWrapper, WikipediaAPIWrapper
    from common.langchain_imports import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
    from common.langchain_imports import initialize_agent, AgentType
    from common.langchain_imports import StreamlitCallbackHandler
    from common.config import load_dotenv
    from common.langchain_imports import Tool
    import os
    load_dotenv()

    ## Arxiv and wikipedia Tools
    arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

    wiki_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)

    search = DuckDuckGoSearchRun(name="Search")

    def safe_duckduckgo_run(query: str):
        try:
            return search.run(query)
        except Exception as e:
            return f"❌ Search failed: {e}"

    safe_search = Tool.from_function(
        func=safe_duckduckgo_run,
        name="SafeSearch",
        description="Search tool with error handling using DuckD rr3s4rtttuckGo"
    )

    st.subheader("🔎 LangChain - Chat with Search")

    api_key = os.getenv('GROQ_API_KEY')

    if "messages" not in st.session_state:
        st.session_state["messages"]=[
            {
                "role": "assistant",
                "content": "Hi, I'm a ChatBot, who can search the web. How can I help you?"
            }
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])

    if prompt:=st.chat_input(placeholder="What is machine learning?"):
        st.session_state.messages.append({
                "role":"user",
                "content": prompt
            }
        )
        st.chat_message("user").write(prompt)

        llm=ChatGroq(groq_api_key=api_key, model="Llama3-8b-8192", streaming=True)

        tools=[arxiv, safe_search, wiki]

        search_agent=initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)

        with st.chat_message("assistant"):
            st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            response=search_agent.run(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({"role": "assistant", "content": response})

            st.write(response)
