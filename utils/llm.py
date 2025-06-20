from common.langchain_imports import ChatGroq
import os 

from crewai_tools import YoutubeChannelSearchTool

from crewai import LLM

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('GROQ_API_KEY')

llm=ChatGroq(groq_api_key=api_key, model="Gemma2-9b-It", streaming=True)

crew_llm = LLM(
    model="llama3-8b-8192",  # Fast and supported
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)