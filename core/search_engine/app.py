import sys
import os
import traceback
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from common.langchain_imports import (
    initialize_agent, AgentType,
    ArxivQueryRun, WikipediaQueryRun,
    ArxivAPIWrapper, WikipediaAPIWrapper,
    StreamlitCallbackHandler, Tool,
    DuckDuckGoSearchRun,
    LLMChain, PromptTemplate
)
from utils.llm import llm


# --------------------------- #
#  🔧 HELPER FUNCTIONS
# --------------------------- #

def safe_search(query: str) -> str:
    """Safely perform DuckDuckGo search with fallback messages."""
    search = DuckDuckGoSearchRun(name="Search")
    try:
        result = search.run(query)
        return result or "No results found."
    except Exception as e:
        return f"❌ Search failed: {e}"


def build_tools():
    """Initialize all tools with clear names and safe defaults."""
    try:
        arxiv_tool = ArxivQueryRun(
            api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=400)
        )
        wiki_tool = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=400)
        )
        search_tool = Tool(
            name="Web Search",
            description="🌐 Searches the web safely using DuckDuckGo.",
            func=lambda q: safe_search(q)
        )

        # Friendly tool names and descriptions
        arxiv_tool.name = "📚 Arxiv (Research Papers)"
        arxiv_tool.description = "Use this to find academic or scientific papers from Arxiv."

        wiki_tool.name = "📖 Wikipedia (General Knowledge)"
        wiki_tool.description = "Use this to fetch general knowledge or definitions."

        return [arxiv_tool, wiki_tool, search_tool]
    except Exception as e:
        st.error(f"Tool initialization failed: {e}")
        return []


def initialize_chat_state():
    """Initialize session state variables for conversation."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Hi, I’m your AI Search Assistant! "
                    "I can search the **web**, **Wikipedia**, or **Arxiv (Research Papers)** to give you detailed, human-like answers. "
                    "Ask me anything!"
                )
            }
        ]


def generate_explained_response(raw_answer: str, query: str) -> str:
    """
    Feed the raw agent result into the LLM to make it more descriptive,
    structured, and conversational — like ChatGPT.
    """
    explain_template = PromptTemplate(
        input_variables=["query", "raw_answer"],
        template=(
            "You are an AI assistant that explains information in a detailed, ChatGPT-like tone.\n"
            "Below is the raw factual answer obtained from various search tools.\n"
            "Your task is to rewrite it into a clear, insightful, and human-style explanation.\n\n"
            "Query: {query}\n\n"
            "Raw Information:\n{raw_answer}\n\n"
            "Now provide a final, descriptive answer with context, reasoning, and smooth flow."
        )
    )

    chain = LLMChain(llm=llm, prompt=explain_template)
    explained_response = chain.run(query=query, raw_answer=raw_answer)
    return explained_response.strip()


# --------------------------- #
#  💬 MAIN APP LOGIC
# --------------------------- #

def run_search_engine():
    st.title("🔎 AI Search Assistant")
    st.caption("Powered by **LangChain**, **DuckDuckGo**, **Wikipedia**, and **Arxiv (Research Papers)**")

    initialize_chat_state()

    # Chat display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # User input
    prompt = st.chat_input("Ask me anything about AI, science, or current events...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching and analyzing..."):
                try:
                    tools = build_tools()
                    if not tools:
                        st.error("No tools available to process your query.")
                        return

                    search_agent = initialize_agent(
                        tools=tools,
                        llm=llm,
                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        handle_parsing_errors=True,
                        verbose=False,
                    )

                    callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

                    # Step 1 — Agent retrieves the raw answer
                    raw_response = search_agent.run(prompt, callbacks=[callback])

                    # Step 2 — Feed that to LLM for refinement
                    explained_response = generate_explained_response(raw_response, prompt)

                    # Step 3 — Show final detailed answer
                    st.session_state.messages.append({"role": "assistant", "content": explained_response})
                    st.markdown(explained_response)

                except Exception as e:
                    st.error("⚠️ Something went wrong while generating the response.")
                    st.exception(e)
                    st.text(traceback.format_exc())