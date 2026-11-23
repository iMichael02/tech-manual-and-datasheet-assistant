# Deployment

## Streamlit web UI & Community Cloud
`apps/streamlit_app.py` is a chat UI over the same pipeline (per-visitor history, citations in an expander, sidebar toggle for the
no-RAG baseline, optional `APP_PASSWORD` gate). Verified with Streamlit's `AppTest` for: Ollama backend, extractive fallback,
out-of-scope refusal, missing-API-key error, password gate.

**Streamlit Cloud cannot reach your local Ollama.** Use Claude there (set `ANTHROPIC_API_KEY`), or the app degrades to the
extractive backend. Steps:
1. Put the project in a GitHub repo (`.gitignore` already excludes the zip, secrets and caches). `data/index/chunks.json` must be
   committed (the app loads it; it doesn't re-ingest). The PDFs are only needed to rebuild the index. The index contains the
   datasheets' full text, so **use a private repo** unless you're sure you may redistribute them.
2. share.streamlit.io -> *New app* -> pick the repo, main file `apps/streamlit_app.py`. `requirements.txt` (`-e .[app]`, which
   installs `pyproject.toml`'s package + Streamlit) and `packages.txt` (poppler-utils) are picked up automatically.
   Not yet verified on Cloud: if the editable install is rejected, use `.[app]` in `requirements.txt` and set the secret/env
   `DATASHEET_ASSISTANT_ROOT` to the repo checkout path (the data/index folders are looked up there).
3. App settings -> *Secrets*: paste `.streamlit/secrets.toml.example` with your real `ANTHROPIC_API_KEY` and an `APP_PASSWORD`.
   Without the password, anyone with the URL spends your API quota.

## Open WebUI tool server (`apps/tool_server.py`)
Exposes the retriever + scope gate as an OpenAPI tool so an Open WebUI chat model can call it. Only retrieval lives here; the
chat model writes the answer from the returned numbered excerpts (document + page), and `in_scope=false` tells it to refuse.
```bash
pip install -e ".[server]"
TOOL_SERVER_API_KEY=choose-a-key uvicorn tool_server:app --app-dir apps --port 8000   # key optional; /docs shows the API
```
In Open WebUI: *Settings → Integrations → Tool Servers → add* `http://localhost:8000` (from Docker: `http://host.docker.internal:8000`),
auth = Bearer + the same key; then enable the **search_datasheets** tool for your model (Workspace → Models) or per chat.
Tool descriptions in the OpenAPI spec tell the model to always search first, cite `[n]` and not guess.

Caveats: the tool-calling path is **not covered by `eval --generation`** (that measures our own pipeline, which always retrieves).
Open WebUI's docs say models under ~30B often emit malformed tool calls even in Native mode, so don't expect `llama3.1:8b` to
match the 29/29; use a larger or hosted model and spot-check the questions from `src/assistant/evaluation/questions.json`. `/health` is unauthenticated.
