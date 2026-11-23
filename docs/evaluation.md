# Evaluation, approach & limitations

## 1. Problem & success criteria
Engineers/technicians need exact specs (addresses, ranges, timings) and procedures from dense PDFs.
A wrong number is worse than no answer, so the priorities are *faithfulness* and *knowing when not to answer*.

| Criterion | Metric | Target | Measured (offline) |
|---|---|---|---|
| Right document retrieved | hit@3 | ≥ 95% | **100%** |
| Evidence passage in what the LLM sees | ev@5 | ≥ 85% | **83%** (misses below) |
| Refuse out-of-scope / unknown parts | refusal rate | 100% | **100%** (7 q) |
| Don't refuse valid questions | false-refusal rate | 0% | **0%** |
| Answer correctness (keyword-graded) | `eval --generation` | ≥ 90% | **100%** (29/29, `llama3.1:8b`, single run; graders partly loosened, see §3) |
| Every answer cites doc + page | manual / test | 100% | enforced by prompt + `test_answer_carries_page_citations` |


## 2. Approach: RAG (chosen) vs. alternatives
| Method | Verdict |
|---|---|
| Pre-prompting only | Model would answer from memory → plausible-but-wrong specs, no citations. Used only as the *system prompt* (grounding rules). |
| **RAG** (chosen) | Documents change per school/product; answers must be traceable to a page; no training needed. |
| Fine-tuning / LoRA | Teaches style, not facts; goes stale, can't cite, needs GPU + curated Q&A. Not worth it for 14 documents. Possible later to improve format adherence. |

Pipeline (`src/assistant/`): `pdftotext` (reading order – the PDFs are two-column; pages detected as tables are
re-extracted with `-layout` so a row's label, value and unit stay together) → 180-word chunks with 40-word overlap,
tagged with doc/page/part aliases → **BM25** → **part-number routing** (a question naming a known part is restricted
to that part's documents; aliases fix misnamed files like `1893_datasheet.pdf` = MPL3115A2) → **scope gate**
(unknown part-like token such as BME680, or low best-chunk IDF coverage ⇒ refuse) → grounded prompt with numbered
excerpts → answer with `[n]` citations.

Why BM25 rather than embeddings: datasheet questions hinge on exact tokens (`0x76`, `DS18B20`, `I2C`) and BM25 is
dependency-free, deterministic and fast. The cost is vocabulary mismatch (below).


## 3. Evaluation (`src/assistant/evaluation/questions.json`, `run.py`)
29 answerable questions (with gold document + evidence regex, verified to exist in the corpus) and 7 out-of-scope ones.

```
config                  hit@1  hit@3  hit@5   ev@1   ev@3   ev@5    MRR
BM25, no routing          83%    97%   100%    41%    59%    72%   0.53
BM25 + part routing      100%   100%   100%    62%    79%    83%   0.69
```
Part routing is the biggest single win (+11 pts ev@5, MRR 0.53→0.69).
End-to-end (`llama3.1:8b` via Ollama, temperature 0): **29/29 keyword-correct, 7/7 correct refusals**.

How these numbers came about (so they aren't over-read):
- **Earlier README figures (ev@5 86%, 26/29) were stale/strict.** The index predated a `catalog.py` fix (`DHT22.pdf` is an
  older AM2303 revision, not a DHT22/AM2302 sheet), and three graders were too strict. After re-ingesting, ev@5 is 83%.
- **Gold-set fixes:** the AM2302 supply question no longer lists `DHT22.pdf` (3.3–6 V) as a valid source, and "sensing
  period of the DHT22 / AM2302" became "…of the AM2303" (`Average: 2s` exists only in `DHT22.pdf`). MAX31855 evidence now
  requires the *clean* resolution text (`14-Bit, 0.25` or the table row), and its answer grader accepts `0.25` alone.
  The DS18B20 data-line grader accepts "parasit|data line". The 29/29 therefore partly reflects grader loosening.
- **Real bug found and fixed:** the MAX31855 resolution answer was wrong (0.0625 °C, the *cold-junction* resolution). Cause:
  `pdftotext` reading order scattered the spec-table labels from their values, so the model paired them wrongly. Fix: table
  pages use `-layout` (see `src/assistant/ingest.py`). Verified on that question; no retrieval metric detects this class of error,
  so spot-check table-derived answers by hand.
- Single run per configuration; small differences (±1 question) are within noise. A model change can move the result.

Remaining retrieval misses (5): "several sensors share one bus", BMP280 pressure range, TMP006 sensing element,
Rigol "automatic setup" (vocabulary gaps), and AM2303 supply voltage (rank 6: `DHT22.pdf` and the AM2303 PDF are both
routed to AM2303 and compete in a small pool). The BMP280 SDO question is rank 4.

**RAG vs. no retrieval** (`python -m assistant --backend ollama eval --generation --norag`, `llama3.1:8b`, same questions):

| Mode | Keyword-correct | Correct refusals |
|---|---|---|
| Pre-prompt only (no retrieval; prompt lists the covered documents) | 15/29 (52%) | 7/7 |
| **RAG** | **29/29 (100%)** | 7/7 |

Of the 14 baseline failures, 10 are confidently wrong specs with no citation (e.g. MPL3115A2 I2C address 0x5C, AM2302/AM2303
supply "3.0–3.6 V", BMP280 range "250–1050 mbar", DS18B20 "needs an external supply") and 4 are over-refusals of answerable
questions. Caveats: the baseline's perfect refusals are easy because its prompt names the covered documents (the RAG
scope gate is stricter, not smarter); it is one run on a small 8B model, and a stronger model would know more datasheet
facts from memory, so re-run with `--backend anthropic` before generalising. The gap that matters is the *wrong-but-confident* answers.


## 4. Known limitations / next steps
- **Vocabulary gaps** (the 5 retrieval misses): "several sensors share one bus" vs "unique 64-bit serial code";
  "automatic setup" vs "AUTO button". Fix: hybrid BM25 + dense embeddings, or LLM query rewriting.
- **Tables:** table pages are re-extracted with `pdftotext -layout` (heuristic: ≥8 rows and ≥30% of lines with ≥3
  columns), which fixed the label/value pairing, but multi-line labels are still split across rows. `pdfplumber` would be
  more robust. Figures and timing diagrams are not indexed (no OCR/vision). `NC` is normalised to `°C` at ingest.
- **Conflicting sources are in the data on purpose:** `DHT22.pdf` is really an older AM2303 revision (3.3–6 V, 20 m)
  contradicting the AM2303 PDF (3.3–5.5 V, 100 m). The prompt tells the model to cite each source and flag the
  disagreement instead of picking one silently. Check this when you run the generation eval.
- Scope gate thresholds (`min_coverage=0.55`) were tuned on a small set; re-tune when documents are added.
- The six `out*.html`/`toto*.html` files in the data folder are Tika output for a BMA150 datasheet that is not
  in the folder; they are ignored by ingestion.
- Adding a document: drop the PDF in `data/datasheets/`, add a line to `src/assistant/catalog.py`, re-run `ingest`
  (the index stores catalog metadata, so it goes stale whenever `catalog.py` changes).

