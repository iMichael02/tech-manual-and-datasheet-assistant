# Technical Manual & Datasheet Assistant

Course project (EPF Gen-AI): a domain chatbot that answers technical questions about the sensors and instruments in
`data/datasheets/`, citing document and page. The deliverable is the assistant plus its evidence: stated success criteria,
an evaluation, and a comparison of implementation methods (pre-prompting, RAG, fine-tuning/LoRA). RAG is the chosen method.

**Faithfulness outranks coverage.** A wrong number is worse than a refusal, so every change is judged on two things: answers
stay grounded in the excerpts with `[n]` citations, and out-of-scope questions are refused.

## Verify a change
- `pytest` (unit tests), then `python3 -m assistant eval` (offline retrieval + refusal metrics). Compare against the numbers in `docs/evaluation.md`.
- Anything touching prompts, retrieval or ingest also needs `python3 -m assistant --backend ollama eval --generation --norag` (local Ollama, `llama3.1:8b` pulled).
- Setup: `pip install -e ".[app,server,dev]"` (the package lives in `src/`; commands are in `README.md`).

## Gotchas the code does not show
- **The index is derived.** `data/index/chunks.json` bakes in `catalog.py` metadata and the extracted text. Re-run `python3 -m assistant ingest` after editing `catalog.py` or `ingest.py`; a stale index once produced misleading metrics.
- **File names mislead.** `DHT22.pdf` is an older AM2303 datasheet (3.3–6 V) that disagrees with the AM2302/AM2303 PDFs; `1893_datasheet.pdf` is the MPL3115A2. The disagreement is deliberate test data: answers cite each source and flag the conflict.
- **Tables extract badly.** Table pages use `pdftotext -layout`, prose uses reading order. No metric detects a garbled table (evidence regexes match garbled text too), so read table-derived answers by hand.
- **Triage a failing eval question** as retrieval, generation or grader before changing code. Grading is keyword regex; keep evidence regexes specific to the clean text.
- **The scope gate is tuned on a small set** (`min_coverage`). Re-run the eval after adding documents.
- **PDFs stay out of git** (`data/datasheets/*` is ignored), yet the committed index holds their text: keep the repo private.
- **Streamlit Cloud cannot reach local Ollama.** Hosted runs use Claude (`ANTHROPIC_API_KEY`) or fall back to extractive mode.

## Adding a document
Drop the PDF in `data/datasheets/`, add its entry to `src/assistant/catalog.py`, run `ingest`, add gold questions to
`src/assistant/evaluation/questions.json`; done when the eval's gold-set sanity line shows every item verifiable in the corpus.

## Further reading
- `docs/evaluation.md`: before changing retrieval, prompts or metrics (results, limits, open misses).
- `docs/deployment.md`: before touching `apps/`, `requirements.txt` or packaging.
