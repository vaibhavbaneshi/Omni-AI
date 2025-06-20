from common.langchain_imports import (
    WikipediaAPIWrapper, Tool, LLMMathChain
)

from utils.llm import crew_llm

from crewai_tools import YoutubeChannelSearchTool

##Initialize tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find the vatious information on the topics mentioned"
)

##Initialize Math Tool
math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tools for answering math related questions. Only input mathematical expression need to bed provided"
)

# Initialize the tool with a specific Youtube channel handle to target your search

yt_tool = YoutubeChannelSearchTool(youtube_channel_handle='@StudyIQEducationLtd', llm=crew_llm)