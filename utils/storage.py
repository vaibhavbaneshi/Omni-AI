# utils/storage.py
import os, json, uuid
from configs.config import JOBS_DIR, OUTPUTS_DIR, CACHE_DIR

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

def save_job_payload(payload: dict) -> str:
    job_id = str(uuid.uuid4())
    path = os.path.join(JOBS_DIR, f"{job_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return job_id

def list_job_files():
    return sorted([f for f in os.listdir(JOBS_DIR) if f.endswith(".json")])

def read_job(job_id: str) -> dict:
    path = os.path.join(JOBS_DIR, f"{job_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_output(job_id: str, filename: str, content: str|bytes):
    out_dir = os.path.join(OUTPUTS_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    mode = "wb" if isinstance(content, (bytes, bytearray)) else "w"
    with open(path, mode) as f:
        f.write(content)
    return path

import os, json
from configs.config import JOBS_DIR

def save_job_file(job: dict):
    """
    Save job JSON safely. Appends a unique ID if needed.
    """
    os.makedirs(JOBS_DIR, exist_ok=True)
    job_id = job.get("job_id")
    if not job_id:
        raise ValueError("Job must have a unique 'job_id' field")

    job_path = os.path.join(JOBS_DIR, f"{job_id}.json")

    # Ensure we don't overwrite existing job file
    counter = 1
    original_path = job_path
    while os.path.exists(job_path):
        job_path = original_path.replace(".json", f"_{counter}.json")
        counter += 1

    with open(job_path, "w") as f:
        json.dump(job, f, indent=2)
    
    return job_path