from common.langchain_imports import ChatGroq
import os 

api_key = os.getenv('GROQ_API_KEY')

llm=ChatGroq(groq_api_key=api_key, model="Gemma2-9b-It", streaming=True)