# Multi-Model Evaluation (v0)

Classy Streamlit UI + SQLite-backed evaluation runner for comparing LLMs on a dashboard summarization task.

## Run  locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
export $(cat .env | xargs)

streamlit run app.py

