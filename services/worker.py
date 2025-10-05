import time, os, json
from utils.storage import list_job_files, read_job, write_output
from services.train_logic import run_training
from utils.logger import setup_logger
from configs.config import JOBS_DIR

logger = setup_logger("worker")

def poll_and_run(interval=5):
    logger.info("Worker started; polling jobs dir %s", JOBS_DIR)
    while True:
        jobs = sorted(list_job_files(), key=os.path.getmtime)  # newest first
        if not jobs:
            time.sleep(interval)
            continue

        # Only process the latest job
        latest_job_file = jobs[-1]  # newest file
        job_id = latest_job_file.replace(".json", "")
        job = read_job(job_id)

        # Skip if already running or done
        if job.get("status") in ("running", "done"):
            time.sleep(interval)
            continue

        # Mark as running
        job["status"] = "running"
        with open(os.path.join(JOBS_DIR, latest_job_file), "w") as f:
            json.dump(job, f, indent=2)

        try:
            logger.info("Starting job %s", job_id)
            out_dir = run_training(
                job_id=job_id,
                jsonl_path=job["jsonl_path"],
                base_model=job.get("base_model"),
                use_qlora=job.get("use_qlora", False),
                epochs=job.get("epochs"),
                batch=job.get("batch"),
                lr=job.get("lr"),
                max_length=job.get("max_length")
            )
            job["status"] = "done"
            job["out_dir"] = out_dir
            logger.info("Job %s done", job_id)
        except Exception as e:
            logger.exception("Job failed %s", job_id)
            job["status"] = "failed"
            job["error"] = str(e)

        with open(os.path.join(JOBS_DIR, latest_job_file), "w") as f:
            json.dump(job, f, indent=2)

        # Optional: Clean other old queued jobs
        for old_job_file in jobs[:-1]:
            try:
                old_job_path = os.path.join(JOBS_DIR, old_job_file)
                os.remove(old_job_path)
                logger.info("Removed old job %s", old_job_file)
            except Exception as e:
                logger.warning("Failed to remove %s: %s", old_job_file, e)

        time.sleep(interval)


if __name__ == "__main__":
    poll_and_run()