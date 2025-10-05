import os, json, time
import streamlit as st
from configs.config import JOBS_DIR
from utils.storage import save_job_file  # hypothetical helper to save job
import uuid

def run_fine_tune():
    st.subheader("🧪 Fine-Tune LLM")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload JSONL dataset", type=["jsonl"])

    # Base model selection
    base_model = st.selectbox("Base Model", ["meta-llama/Llama-2-7b-hf"])

    epochs = st.number_input("Epochs", value=3, min_value=1)
    batch = st.number_input("Batch Size", value=4, min_value=1)
    lr = st.number_input("Learning Rate", value=2e-5, format="%.6f")
    max_length = st.number_input("Max Length", value=1024)

    if st.button("🚀 Start Fine-Tune Job"):
        if uploaded_file is None:
            st.warning("Please upload a dataset first!")
        else:
            # Save uploaded file
            file_path = os.path.join(JOBS_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Create job
            job_id = str(uuid.uuid4())
            job = {
                "job_id": job_id,
                "jsonl_path": file_path,
                "base_model": base_model,
                "epochs": epochs,
                "batch": batch,
                "lr": lr,
                "max_length": max_length,
                "status": "queued"
            }
            save_job_file(job)  # write JSON in JOBS_DIR

            # Show spinner until job finishes
            with st.spinner("Fine-tuning in progress..."):
                finished = False
                while not finished:
                    job_path = os.path.join(JOBS_DIR, f"{job_id}.json")
                    with open(job_path, "r") as f:
                        job_status = json.load(f)
                    if job_status.get("status") in ("done", "failed"):
                        finished = True
                    else:
                        time.sleep(5)  # poll every 5 sec

            # Display final result
            st.success(f"Job Finished! Status: {job_status['status']}")
            st.write(f"Output Directory: {job_status.get('out_dir','N/A')}")