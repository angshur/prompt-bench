import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from db import init_db, fetch_runs
from engine import run_eval

load_dotenv()
st.set_page_config(page_title="Multi-Model Evaluation", layout="wide")
init_db()

st.markdown(
    """<style>
:root {
  --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.08);
  --stroke: rgba(255,255,255,0.10);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.65);
  --good: #34d399;
  --bad: #fb7185;
}

.stApp {
  background: radial-gradient(1200px 800px at 10% 0%, rgba(110,231,255,0.15), transparent 55%),
              radial-gradient(900px 650px at 100% 0%, rgba(167,139,250,0.18), transparent 55%),
              linear-gradient(180deg, #070b12 0%, #0b1220 60%, #070b12 100%) !important;
  color: var(--text) !important;
}

h1, h2, h3, h4, h5, h6, p, label, div, span { color: var(--text) !important; }

.block-container { padding-top: 2.2rem; padding-bottom: 2.2rem; max-width: 1200px; }

.card {
  background: var(--card);
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 18px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

.card-soft {
  background: var(--card2);
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 16px 16px;
}

.kpi {
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--stroke);
  border-radius: 14px;
  padding: 12px 12px;
}

.badge {
  display: inline-block;
  padding: 3px 10px;
  border: 1px solid var(--stroke);
  border-radius: 999px;
  background: rgba(255,255,255,0.04);
  color: var(--muted);
  font-size: 12px;
  margin-right: 8px;
}

.badge-ok { color: var(--good); border-color: rgba(52,211,153,0.35); background: rgba(52,211,153,0.08); }
.badge-warn { color: #fbbf24; border-color: rgba(251,191,36,0.35); background: rgba(251,191,36,0.08); }
.badge-bad { color: var(--bad); border-color: rgba(251,113,133,0.35); background: rgba(251,113,133,0.08); }

.stButton>button {
  border-radius: 12px;
  padding: 0.6rem 1.0rem;
  border: 1px solid rgba(255,255,255,0.18);
  background: linear-gradient(90deg, rgba(110,231,255,0.22), rgba(167,139,250,0.22));
  color: white;
  font-weight: 700;
}

.stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] > div,
.stFileUploader section {
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: rgba(255,255,255,0.04) !important;
}

hr { border-color: rgba(255,255,255,0.10); }
</style>""",
    unsafe_allow_html=True,
)

MODEL_CATALOG = {
    "OpenAI": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    "Gemini": ["gemini-1.5-pro", "gemini-1.5-flash"],
    "Claude/Anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
    "xAI (Grok)": ["grok-2"],
    "Llama": ["llama-3.1-70b", "llama-3.2-8b", "llama-3.2-1b"],
}

l, r = st.columns([2.2, 1])
with l:
    st.markdown("## Query Multi-Model Evaluation")
    st.markdown('<span class="badge">v0</span><span class="badge">task: dashboard_summarization</span>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color: rgba(255,255,255,0.65); margin-top:6px;">'
        "Compare outputs + latency/tokens across providers using the same uploaded dashboard export."
        "</div>",
        unsafe_allow_html=True,
    )

with r:
    st.markdown(
        '<div class="card-soft">'
        '<div style="font-weight:700; margin-bottom:6px;">What gets logged</div>'
        '<div style="color: rgba(255,255,255,0.70); font-size:13px; line-height:1.45;">'
        "TestDataID (content hash), prompt_hash, model/version, latency, tokens, schema flags, rubric scores (optional judge)."
        "</div></div>",
        unsafe_allow_html=True,
    )

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Input")

c1, c2 = st.columns(2)
with c1:
    provider = st.selectbox("Model", options=list(MODEL_CATALOG.keys()), index=0)
with c2:
    version = st.selectbox("Version", options=MODEL_CATALOG[provider], index=0)

uploaded = st.file_uploader("Upload Test Data (PDF/CSV/XLSX)", type=["pdf", "csv", "xlsx"])
prompt = st.text_area(
    "Enter Prompt / Text",
    height=170,
    placeholder="Paste your summarization instructions here... (v0 = free text)",
)
who = st.text_input("who (optional)", value="")

cols = st.columns([1, 2.6, 1.4])
with cols[0]:
    submit = st.button("Submit", type="primary", disabled=(uploaded is None or not prompt.strip()))
with cols[1]:
    st.caption("Runs evaluation and returns the output + scoring + metrics.")
st.markdown("</div>", unsafe_allow_html=True)

if submit:
    run_id = run_eval(
        who=who.strip() or None,
        provider=provider,
        model_version=version,
        filename=uploaded.name,
        file_bytes=uploaded.getvalue(),
        user_prompt=prompt,
    )
    st.success(f"Submitted. run_id={run_id}")

st.markdown("---")

runs = fetch_runs(limit=50)

st.markdown("### Latest Result")
if not runs:
    st.info("No runs yet. Upload a dashboard export, add a prompt, and Submit.")
else:
    latest = runs[0]
    status = latest["status"]

    badge = "badge-warn"
    if status == "SUCCESS":
        badge = "badge-ok"
    elif status == "ERROR":
        badge = "badge-bad"

    st.markdown(
        f'<span class="badge {badge}">status: {status}</span>'
        f'<span class="badge">provider: {latest["provider"]}</span>'
        f'<span class="badge">model: {latest["model_version"]}</span>'
        f'<span class="badge">TestDataID: {latest["test_data_id"]}</span>'
        f'<span class="badge">prompt_hash: {(latest["prompt_hash"] or "")[:12]}</span>',
        unsafe_allow_html=True,
    )

    if status == "ERROR":
        st.error(latest.get("error_message") or "Unknown error")
    elif status != "SUCCESS":
        st.info(f"Latest run is {status}. Refresh if needed.")
    else:
        left, right = st.columns([2.2, 1.0])

        with left:
            st.markdown('<div class="card-soft">', unsafe_allow_html=True)
            st.markdown("**Generated output text**")
            st.write(latest["output_text"])
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card-soft">', unsafe_allow_html=True)
            st.markdown("**System Metrics**")
            flags = json.loads(latest["numeric_flags_json"] or "{}")
            st.markdown(
                f"""
<div class="kpi"><div style="color: rgba(255,255,255,0.65); font-size:12px;">latency_ms</div>
<div style="font-size:20px; font-weight:700;">{latest["latency_ms"]}</div></div>

<div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-top:10px;">
  <div class="kpi"><div style="color: rgba(255,255,255,0.65); font-size:12px;">tokens_in</div>
  <div style="font-size:18px; font-weight:700;">{latest["tokens_in"]}</div></div>
  <div class="kpi"><div style="color: rgba(255,255,255,0.65); font-size:12px;">tokens_out</div>
  <div style="font-size:18px; font-weight:700;">{latest["tokens_out"]}</div></div>
</div>

<div class="kpi" style="margin-top:10px;">
  <div style="color: rgba(255,255,255,0.65); font-size:12px;">validation</div>
  <div style="margin-top:4px;">
    <span class="badge {'badge-ok' if latest['schema_valid'] else 'badge-warn'}">schema_valid: {bool(latest['schema_valid'])}</span>
    <span class="badge {'badge-ok' if latest['required_sections_present'] else 'badge-warn'}">required_sections_present: {bool(latest['required_sections_present'])}</span>
  </div>
  <div style="margin-top:8px; color: rgba(255,255,255,0.70); font-size:13px;">
    numeric flags: {flags}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card-soft" style="margin-top:12px;">', unsafe_allow_html=True)
            st.markdown("**Quality Scoring (rubric)**")
            score_map = json.loads(latest["score_json"] or "{}")
            st.write({"overall": latest["score_overall"], **score_map})
            st.caption("Set JUDGE_MODEL_PROVIDER + JUDGE_MODEL to enable rubric scoring.")
            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Run History (Audit Log)")

def to_history_df(rows):
    out = []
    for r in rows:
        score_map = json.loads(r["score_json"] or "{}")
        out.append(
            {
                "who": r["who"],
                "when": r["created_at"],
                "provider": r["provider"],
                "model/version": r["model_version"],
                "prompt_id": r["prompt_id"],
                "prompt_hash": (r["prompt_hash"] or "")[:12],
                "TestDataID": r["test_data_id"],
                "overall": r["score_overall"],
                "Accuracy": score_map.get("Accuracy"),
                "Clarity": score_map.get("Clarity"),
                "Professionalism": score_map.get("Professionalism"),
                "Insightfulness": score_map.get("Insightfulness"),
                "Next Steps": score_map.get("Next Steps"),
                "Assumption Labeling": score_map.get("Assumption Labeling"),
                "Tone Alignment": score_map.get("Tone Alignment"),
                "latency_ms": r["latency_ms"],
                "tokens_in": r["tokens_in"],
                "tokens_out": r["tokens_out"],
                "status": r["status"],
            }
        )
    return pd.DataFrame(out)

df = to_history_df(runs) if runs else pd.DataFrame()
st.dataframe(df, use_container_width=True, height=420)

