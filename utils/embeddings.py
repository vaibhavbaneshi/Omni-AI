from common.langchain_imports import HuggingFaceEmbeddings
import os

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

embeddings=HuggingFaceEmbeddings(model_name="Sybghat/all-MiniLM-L6-v2-finetuned-squad")