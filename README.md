# LexiSyncAI - Evidence-Grounded Legal Document Analyst

**DA3 Deep Learning & GenAI final project** - Individual project aligned with the supplied course assignment. A Streamlit RAG prototype that searches legal PDFs with BM25 + MiniLM embeddings + reciprocal rank fusion (RRF) + cross-encoder reranking, asks a local Ollama model (or optional API model) to answer with page/section-level citations, verifies citation IDs and literal quotations, and abstains when evidence is inadequate.

> **Status:** Source code and 5 automated offline unit tests are implemented and passed in the preparation environment. The large pretrained neural weights, UI end-to-end inference, 2–5 real legal PDF ingestion, and measured results have **not been executed or validated in that environment**. They require installation and tests on your laptop. Never claim the full application or evaluation is validated until you run them yourself.

## System Architecture

![LexiSyncAI Architecture](architecture.png)

## Quick start on macOS / VS Code

**Simplest route:** `bash setup_mac.sh`, then `source .venv/bin/activate`, download the Ollama model, then `./run_mac.sh`. The detailed manual commands are below.

1. Install **Python 3.12** (recommended), Git, and [Ollama](https://ollama.com/download). Python 3.13 may work but if package installation fails use 3.12. You use **VS Code's Terminal**, not a Jupyter notebook/kernel.
2. Open this `LexiSyncAI` folder in VS Code and open **Terminal > New Terminal**.
3. Run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b
# Ensure the Ollama app is running, or in a second terminal use: ollama serve
python -m streamlit run app.py
```

If `python3.12` is missing but your existing Python is compatible, use `python3 -m venv .venv`. The terminal must show `(.venv)`; `which python` should point inside the project's `.venv/bin/`. On Apple silicon, let sentence-transformers use its defaults (CPU is a safe fallback); pretrained models download on the **first index**, so arrange reliable internet beforehand.

4. In the browser, create a case, upload **2–5 real PUBLIC PDFs** (sources in `docs/SOURCES.md`), and click **Index documents / pre-warm models**. A normal first run can be slower because weights download; subsequent cached runs are faster.
5. Enter a factual question and click **Answer and show sources**. Read the cited evidence and select an E-number to see the original PDF page with an approximate highlighted block. Then ask a two-article/multi-hop question and an unanswerable question. Never rely on the coverage label as calibrated probability.
6. Before the professor's exam, start Ollama, confirm both models have been downloaded, test all three question types, and keep the app running.

**If Ollama is not usable:** choose **OpenAI API** in the sidebar, paste your own API key into the password field, and choose the supported model configured in `app.py`. API use may cost money; it is not included with ChatGPT Plus. Only use public non-confidential files. The application does not store the API key in files.

### Run tests and evaluation

```bash
python -m pytest -q
mkdir -p evaluation/pdfs
# Manually download and add GDPR.pdf, AI_Act.pdf from official sources
python evaluate.py --retrieval-only
python evaluate.py --with-llm --provider Ollama --model qwen2.5:3b
```

Read `evaluation/README.md` **before** reporting numbers. The 18-question test has 15 answerable (factual/multi-hop/multi-document) and 3 unanswerable examples. For complete marks, manually validate answer correctness and citation precision in the exported CSV. Compare dense-only vs hybrid and include recall@k, abstention accuracy and latency measured on the same corpus.

## How the application works

| Component | Technique | Why it is used | Location |
|---|---|---|---|
| UI | Streamlit | case, PDF upload, index, questions, answer/citation pane | `app.py` |
| Ingestion | PyMuPDF position-aware blocks, two-column heuristic, repeating margin removal | retains page provenance and mitigates reading order artifacts | `lexisync/ingest.py` |
| Structure | regex article/clause recognition; paragraph-first splitting | keeps legal context and prevents spanning physical PDF pages | `lexisync/ingest.py` |
| Scan flags | low-text/high-image page detection | warns when input is likely image-only and not searchable | `lexisync/ingest.py` |
| Table parsing | PyMuPDF `find_tables()` | extracts simple tables as separate passages | `lexisync/ingest.py` |
| Lexical retrieval | BM25 Okapi (built here) | works well for exact numbers and section labels | `lexisync/retrieval.py` |
| Semantic retrieval | `all-MiniLM-L6-v2` sentence-transformer | matches similar wording even if terms differ | `lexisync/retrieval.py` |
| Fusion | reciprocal rank fusion (RRF) | combines BM25 and dense ranks robustly | `lexisync/retrieval.py` |
| Reranking | `ms-marco-MiniLM-L-6-v2` cross-encoder | rescoring on query/passage pairs | `lexisync/retrieval.py` |
| Generation | Ollama `qwen2.5:3b` local, or separate OpenAI API | writes concise evidence-bound natural-language claims | `lexisync/generation.py` |
| Verification | IDs + contiguous substring quotations | mechanically rejects fabricated citations; **not semantic entailment** | `lexisync/generation.py` |
| Citation UI | PDF page number, section, highlighted extracted bbox | user can inspect original text in context | `app.py` |
| Evaluation | 18 example questions, recall@k, latency, unanswerable cases; manual CSV | comparison and error analysis | `evaluate.py`, `evaluation/` |

**Pipeline diagram:** `docs/Architecture_One_Page.pdf`, also editable visually in `docs/Architecture_One_Page.svg`.

## User-visible features

* Case name, public PDF uploader, 2–5 PDF validation; optional single-PDF oral-exam mode.
* Displays detected pages, chunks, sections, tables, and likely scanned page numbers.
* Four retrieval options: BM25, dense-only, hybrid, hybrid+reranker.
* Claim-by-claim evidence IDs and verifiable page/section citations, original page image and approximate block highlight.
* Refusal when the model reports insufficient evidence or cites invalid sources/quotes.
* Qualitative `HIGH`, `MEDIUM`, `LOW`, `INSUFFICIENT EVIDENCE`; **these are heuristics, not calibrated model confidence**.
* Measured retrieval/rerank/generation seconds and token counts when the provider supplies them.
* JSON output for manual audit.

## Scope and known limitations (tell the professor)

1. **Real legal accuracy is not guaranteed.** A verbatim quote supports provenance, not necessarily the truth of an inference. An expert must check legal interpretation and legal versions.
2. Digital PDFs are supported; **OCR for scanned pages is not implemented**. Some scanned pages may also be misdetected by the heuristic; verify them.
3. Legal section extraction, two-column reading order, and table extraction are **heuristic**. Complex nested tables, headers spanning columns, and footnotes require review.
4. PDF page numbers are **physical page indices (1-based)**, which can differ from the printed page numbers in legal documents. The citation label makes this explicit.
5. The region highlight surrounds extracted text blocks, not guaranteed exact claim words. No region-level citation-precision benchmark has been performed.
6. Cross-document reconciliation is encouraged in the prompt but **not a validated conflict-detection engine**. The answer may abstain if the model notices conflict.
7. `HIGH/MEDIUM/LOW` are deterministic evidence coverage heuristics, **not statistical confidence**.
8. Large 100+ page PDFs and the first neural model downloads may take time. Measure and plan around real startup latency.
9. The app does not fine-tune an embedding model, train a new LLM, implement OCR/VLM, or deploy publicly. Do not list these as completed features.
10. No answer is generated when provider connection fails; an error is shown instead of a deceptive fallback.

## Architecture and reproducibility details

* Extraction caches are keyed by PDF SHA-256 + extraction mode. Dense vectors keyed by model identifier and text+chunk fingerprint.
* Most dependencies are pinned to exact release versions in `requirements.txt`. The native torch/transformers transitive dependency tree is not fully locked; for strict reproducibility run `python -m pip freeze > resolved_requirements.txt` **after successful installation** and commit the lock file. Record Ollama model digest and Python version.
* Test with `python -m pytest -q` before pushing. Seed non-deterministic operations in any future training code (there is no training here); retrieval algorithms are deterministic given weights/input, while remote LLM generation may vary despite temperature 0.
* Files in `data/cases/`, `data/cache/`, credentials, private PDFs and `.venv` are excluded from Git by `.gitignore`.

## Submission checklist (professor's instructions)

- [ ] Run the UI on real public legal PDFs (2–5) and verify one claim and its citation against the original PDF.
- [ ] Rehearse the **unseen PDF** case, a factual question, one question spanning **two physical pages**, and one truly unanswerable question.
- [ ] Run and validate the evaluation: at least 15 held-out questions, >=3 unanswerable, multi-hop, recall@k across configurations, manual answer correctness and citation precision, abstention behavior and latency.
- [ ] Add actual measured numbers/screenshots to `docs/RESULTS_TO_FILL.md` and presentation. Do not invent values.
- [ ] Record PDF names, dates, hashes, hardware and models in documentation.
- [ ] Push code to **your own GitHub account**, with incremental meaningful commits. Do not fabricate a repository link or commit history.
- [ ] Attach documentation/README and one-page architecture; include GitHub link in the documentation. Upload documentation and presentation to eCampus.
- [ ] Practice the exact 5-minute demo plus 5-minute theory Q&A. See `docs/DEMO_SCRIPT.md` and `docs/THEORY_QA.md`.

**Academic honesty:** This project starter code should be reviewed, understood, tested, modified, and accurately attributed according to your university's AI-tool policy. You need to be able to explain every function and observed result yourself.
