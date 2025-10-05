# utils/data_prep.py
import json, io
from typing import Tuple
from utils.logger import setup_logger

logger = setup_logger("data_prep")

def validate_and_convert_to_jsonl(file_bytes: bytes, filename: str) -> Tuple[bool, str]:
    """
    Accepts CSV/TXT/JSONL as bytes and returns a JSONL string or error.
    Expects instruction-style JSON or CSV with columns: instruction,input,output
    """
    text = file_bytes.decode("utf-8", errors="replace")
    # quick JSONL detection
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # If JSON lines, validate
    try:
        parsed = [json.loads(l) for l in lines if l]
        # validate structure
        for p in parsed:
            if not any(k in p for k in ("instruction","prompt","input")) or "output" not in p:
                logger.warning("JSONL line missing keys: %s", p)
        return True, "\n".join([json.dumps(p, ensure_ascii=False) for p in parsed])
    except Exception:
        # If CSV-like, attempt simple CSV parse
        try:
            import pandas as pd
            df = pd.read_csv(io.StringIO(text))
            # require columns
            required = {"output"}
            if not required.issubset(set(df.columns)):
                return False, "CSV must contain at least 'output' column"
            # map columns
            records = []
            for _, r in df.iterrows():
                rec = {
                    "instruction": r.get("instruction",""),
                    "input": r.get("input",""),
                    "output": r["output"]
                }
                records.append(rec)
            return True, "\n".join([json.dumps(r, ensure_ascii=False) for r in records])
        except Exception as e:
            return False, f"Failed to parse file: {e}"