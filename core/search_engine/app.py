import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common.streamlit_imports import st
from common.langchain_imports import initialize_agent, AgentType, ArxivQueryRun, WikipediaQueryRun, ArxivAPIWrapper, WikipediaAPIWrapper, StreamlitCallbackHandler, Tool, DuckDuckGoSearchRun

from common.config import load_dotenv

from utils.llm import llm

import os
load_dotenv()

def run_search_engine():
    st.subheader("🔎 LangChain - Chat with Search")

    search = DuckDuckGoSearchRun(name="Search")

    # Safe wrapper
    def _safe_search(query: str) -> str:
        try:
            result = search.run(query)
            return result if result else "No results found."
        except Exception as e:
            return f"❌ Search failed: {e}"
        
    arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

    wiki_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)

    safe_search = Tool(
        name="safe_search",
        description="Use this tool to perform safe DuckDuckGo web search",
        func=lambda query: _safe_search(query)
    )

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

        tools=[arxiv, wiki, safe_search]

        search_agent=initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

        with st.chat_message("assistant"):
            st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            response=search_agent.run(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({"role": "assistant", "content": response})

            st.write(response)
