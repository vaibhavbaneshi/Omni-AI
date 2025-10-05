import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

USE_NEO4J = bool(NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD)

# Model / Training defaults
BASE_MODEL = os.getenv("BASE_MODEL", "meta-llama/Llama-2-7b-hf")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "HF")  # placeholder for extensibility

# Local storage
JOBS_DIR = os.getenv("JOBS_DIR", "./jobs")
OUTPUTS_DIR = os.getenv("OUTPUTS_DIR", "./outputs")
CACHE_DIR = os.getenv("CACHE_DIR", "./cache")

# Training defaults
DEFAULT_EPOCHS = int(os.getenv("DEFAULT_EPOCHS", 3))
DEFAULT_BATCH = int(os.getenv("DEFAULT_BATCH", 4))
DEFAULT_LR = float(os.getenv("DEFAULT_LR", 2e-5))
DEFAULT_MAX_LENGTH = int(os.getenv("DEFAULT_MAX_LENGTH", 1024))

# QLoRA/LoRA toggles (used by UI)
ALLOW_QLORA = os.getenv("ALLOW_QLORA", "true").lower() == "true"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_DIR = os.path.join(BASE_DIR, "jobs")  # This is your jobs directory

# Create the folder if it doesn't exist
if not os.path.exists(JOBS_DIR):
    os.makedirs(JOBS_DIR)
