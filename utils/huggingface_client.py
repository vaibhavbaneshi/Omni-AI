from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
load_dotenv()
# Paste your HF token here or set via environment variable
token = os.getenv('HF_TOKEN')

# Pick a chat-supported model
model_id = "HuggingFaceH4/zephyr-7b-beta"  # Example

client = InferenceClient(model=model_id, token=token)