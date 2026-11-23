# Technical Manual & Datasheet Assistant

A domain-specific chatbot that answers technical questions about the sensors and instruments in
`data/datasheets/` (DS18B20, DHT22/AM2302/AM2303, HTU21D, HDC1008, BMP180/280, MPL3115A2,
MAX31855, TMP006, SLHT5, Rigol DS1000E) and **cites the document and page** for every answer.

## Quick start
```bash
sudo apt install poppler-utils
python -m venv venv && source venv/bin/activate
pip install -e ".[app,server,dev]"                 # package (src/assistant) + web UI + tool server + tests
python -m assistant ingest                         # data/datasheets/*.pdf -> data/index/chunks.json (768 chunks)
python -m assistant ask "What is the I2C address of the BMP280 when SDO is tied to VDDIO?"
python -m assistant ask --no-rag "..."             # baseline: same model, no retrieval
python -m assistant chat                           # interactive, keeps short history
python -m assistant eval [--generation [--norag]]  # evaluation suite
streamlit run apps/streamlit_app.py                # web UI
uvicorn tool_server:app --app-dir apps --port 8000 # Open WebUI tool server
pytest                                             # unit tests
```
Re-run `ingest` after editing `src/assistant/catalog.py` or `ingest.py` (the index stores catalog metadata).

Generator backend (`--backend auto|anthropic|ollama|extractive`): `auto` uses Claude if `ANTHROPIC_API_KEY`
is set (model via `ANTHROPIC_MODEL`), else a local Ollama (`OLLAMA_MODEL`, default `llama3.1:8b`), else the
**extractive** fallback, which quotes the best passages verbatim (no LLM, cannot hallucinate, but doesn't synthesize).
Ollama needs the model pulled first (`ollama pull llama3.1:8b`; a missing model gives a 404). `:cloud` models are
proxied to Ollama's servers and can return 402 (sign in / plan or quota); prefer a local model for the 36-call eval.

## Layout
```
src/assistant/              core package: ingest, retrieval (BM25 + part routing + scope gate), generate, pipeline, cli
src/assistant/evaluation/   eval harness + questions.json (gold set)
apps/                       streamlit_app.py (web UI), tool_server.py (OpenAPI tool for Open WebUI)
data/datasheets/            source PDFs
data/index/chunks.json      built index (commit it for deployment)
docs/                       evaluation.md (problem, approach, results, limitations), deployment.md
tests/                      pytest suite
pyproject.toml, requirements.txt, packages.txt   packaging & deployment
```

## Documentation
- [docs/evaluation.md](docs/evaluation.md) — success criteria, RAG vs. alternatives, results (RAG 29/29 vs. no-RAG 15/29), limitations
- [docs/deployment.md](docs/deployment.md) — Streamlit Community Cloud, Open WebUI tool server
