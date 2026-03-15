import os
import json
import uuid
import hashlib
from datetime import datetime
from typing import Tuple, Optional, Dict

import pandas as pd
from pypdf import PdfReader

from db import insert_run, update_run
from providers import call_model
from schema import schema_valid, required_sections_present, numeric_flags

DATA_DIR = "data/testdata"


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def extract_text_from_file(filename: str, file_bytes: bytes) -> str:
    lower = filename.lower()

    if lower.endswith(".pdf"):
        tmp_path = os.path.join(DATA_DIR, f"tmp_{uuid.uuid4().hex}.pdf")
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)
        reader = PdfReader(tmp_path)
        return "\n".join([(p.extract_text() or "") for p in reader.pages])

    if lower.endswith(".csv"):
        import io
        df = pd.read_csv(io.BytesIO(file_bytes))
        return df.head(200).to_csv(index=False)

    if lower.endswith(".xlsx"):
        import io
        df = pd.read_excel(io.BytesIO(file_bytes))
        return df.head(200).to_csv(index=False)

    raise ValueError("Unsupported file type. Use PDF/CSV/XLSX")


def build_task_prompt(user_prompt: str, extracted_data: str) -> str:
    return f"""
TASK: dashboard_summarization

You are given a dashboard export (table/text). Produce a structured summary for an analytics/marketing stakeholder.

REQUIRED OUTPUT SECTIONS (use these exact headings):
1. Executive Summary
2. Key Insights
3. Trends
4. Recommendations
5. Next Steps
6. Assumptions
7. Data Quality Notes

STYLE:
- concise, professional, decision-oriented
- explicitly call out uncertainties as assumptions
- avoid hallucinating metrics not in the data

USER PROMPT:
{user_prompt.strip()}

DASHBOARD EXPORT (may be truncated):
{extracted_data.strip()}
""".strip()


def score_with_judge(output_text: str, input_text: str) -> Tuple[Optional[float], Dict[str, float]]:
    provider = os.getenv("JUDGE_MODEL_PROVIDER")
    model = os.getenv("JUDGE_MODEL")
    if not provider or not model:
        return (None, {})

    judge_prompt = f"""
You are grading a model output for a dashboard summarization task.

Return STRICT JSON only with:
{{
  "Accuracy": 0-10,
  "Clarity": 0-10,
  "Professionalism": 0-10,
  "Insightfulness": 0-10,
  "Next Steps": 0-10,
  "Assumption Labeling": 0-10,
  "Tone Alignment": 0-10
}}

INPUT (for reference):
{input_text[:6000]}

OUTPUT TO GRADE:
{output_text[:6000]}
""".strip()

    r = call_model(provider=provider, model_version=model, prompt=judge_prompt)
    raw = (r.get("output_text") or "").strip()

    try:
        start = raw.find("{")
        end = raw.rfind("}")
        j = json.loads(raw[start : end + 1])
        scores = {k: float(j[k]) for k in j.keys()}
        overall = sum(scores.values()) / max(len(scores), 1)
        return overall, scores
    except Exception:
        return (None, {"judge_parse_error": 1.0})


def run_eval(
    who: Optional[str],
    provider: str,
    model_version: str,
    filename: str,
    file_bytes: bytes,
    user_prompt: str,
) -> str:
    ensure_dirs()

    run_id = uuid.uuid4().hex
    created_at = datetime.utcnow().isoformat()

    test_hash = sha256_bytes(file_bytes)
    test_data_id = f"td_{test_hash[:12]}"

    stored_name = f"{test_data_id}__{filename}"
    stored_path = os.path.join(DATA_DIR, stored_name)
    if not os.path.exists(stored_path):
        with open(stored_path, "wb") as f:
            f.write(file_bytes)

    prompt_id = "adhoc_prompt"
    prompt_hash = sha256_text(user_prompt)

    insert_run(
        {
            "run_id": run_id,
            "created_at": created_at,
            "who": who,
            "task": "dashboard_summarization",
            "provider": provider,
            "model_version": model_version,
            "prompt_id": prompt_id,
            "prompt_hash": prompt_hash,
            "prompt_text": user_prompt,
            "test_data_id": test_data_id,
            "test_data_hash": test_hash,
            "test_data_filename": filename,
            "status": "PENDING",
            "error_message": None,
            "output_text": None,
            "latency_ms": None,
            "tokens_in": None,
            "tokens_out": None,
            "cost_est_usd": None,
            "score_overall": None,
            "score_json": json.dumps({}),
            "schema_valid": 0,
            "required_sections_present": 0,
            "numeric_flags_json": json.dumps({}),
        }
    )

    try:
        extracted = extract_text_from_file(filename, file_bytes)
        final_prompt = build_task_prompt(user_prompt, extracted)

        model_resp = call_model(provider=provider, model_version=model_version, prompt=final_prompt)
        out = model_resp.get("output_text") or ""

        flags = numeric_flags(out)
        req_ok = required_sections_present(out)
        valid = schema_valid(out)

        overall, score_map = score_with_judge(output_text=out, input_text=final_prompt)

        update_run(
            run_id,
            {
                "status": "SUCCESS",
                "output_text": out,
                "latency_ms": int(model_resp["latency_ms"]) if model_resp.get("latency_ms") is not None else None,
                "tokens_in": model_resp.get("tokens_in"),
                "tokens_out": model_resp.get("tokens_out"),
                "cost_est_usd": model_resp.get("cost_est_usd"),
                "score_overall": overall,
                "score_json": json.dumps(score_map),
                "schema_valid": 1 if valid else 0,
                "required_sections_present": 1 if req_ok else 0,
                "numeric_flags_json": json.dumps(flags),
            },
        )
    except Exception as e:
        update_run(run_id, {"status": "ERROR", "error_message": str(e)[:500]})

    return run_id

