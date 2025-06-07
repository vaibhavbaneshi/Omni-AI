from common.langchain_imports import ChatGroq
import os 

api_key = os.getenv('GROQ_API_KEY')

llm=ChatGroq(groq_api_key=api_key, model="Llama3-8b-8192", streaming=True)