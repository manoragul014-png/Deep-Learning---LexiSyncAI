# LexiSyncAI - Evidence-Grounded Legal Document Analyst

**DA3 Deep Learning & GenAI final project** - Individual project aligned with the supplied course assignment. A Streamlit RAG prototype that searches legal PDFs with BM25 + MiniLM embeddings + reciprocal rank fusion (RRF) + cross-encoder reranking, asks a local Ollama model (or optional API model) to answer with page/section-level citations, verifies citation IDs and literal quotations, and abstains when evidence is inadequate.

> **Status:** Source code and 5 automated offline unit tests are implemented and passed in the preparation environment. The large pretrained neural weights, UI end-to-end inference, 2–5 real legal PDF ingestion, and measured results have **not been executed or validated in that environment**. They require installation and tests on your laptop. Never claim the full application or evaluation is validated until you run them yourself.

## System Architecture

![Deep-Learning---LexiSyncAI](Architecture_Diagram.png)

# Installation guide and steps to run the project :

### Step 1: Download and Open the Project
1. Unzip the project into your **Documents** folder.
2. Open **VS Code**.
3. Navigate to **File → Open Folder**.
4. Select the **LexiSyncAI** project folder.

### Step 2: Install Required Software
Make sure the following software is installed:

- **Python 3.12**
- **Ollama**
- **Visual Studio Code (VS Code)**

Install and start the Ollama application before proceeding.

### Step 3: Set Up the Python Environment
Open a new terminal in VS Code using **Terminal → New Terminal**.

Run the following command once:

```bash
bash setup_mac.sh
```

This command automatically:
- Creates a virtual environment (`.venv`) inside the project folder.
- Installs the required Python libraries.

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Python will now use the project's virtual environment.

### Step 4: Download the Local Language Model
Make sure Ollama is running.

Execute:

```bash
ollama pull qwen2.5:3b
```

This downloads the **Qwen 2.5 (3B)** language model for local execution.

**Note:** The embedding model and reranker will download automatically during the first indexing operation. An internet connection is required for the initial setup.

### Step 5: Launch the LexiSyncAI Application
Run:

```bash
./run_mac.sh
```

The LexiSyncAI application should automatically open in your web browser using Streamlit.

**Note:** Streamlit does not require selecting a Jupyter kernel in VS Code.

### Step 6: Upload and Index Legal Documents
1. Open `docs/SOURCES.md`.
2. Download two real, publicly available legal PDF documents, for example:
   - General Data Protection Regulation (GDPR)
   - European Union Artificial Intelligence Act (EU AI Act)
3. Open the LexiSyncAI application.
4. Create a new case.
5. Upload both PDF documents.
6. Click **Index**.

Observe the extracted:
- Pages
- Text chunks
- Document sections
- Scanned-page flags

### Step 7: Test the RAG System
Perform the following tests:

**Test 1: Answer Verification**
- Ask a question with a verifiable answer.
- Select evidence **E1** to view the exact physical PDF page supporting the answer.

**Test 2: Multiple-Page Retrieval**
- Ask a question that requires information from two different physical PDF pages.
- Verify that the system retrieves relevant evidence from both pages.

**Test 3: Hallucination Prevention**
- Ask a question that cannot be answered using the uploaded documents.
- Confirm that the system does not invent unsupported information.

### Step 8: Understand the System Architecture
Open:

`docs/Architecture_One_Page.pdf`

Explain the architecture in the following sequence:

**Ingestion → Retrieval → Reranking → Generation → Verification → UI/Evaluation**

Review and practice the presentation script available at:

`docs/DEMO_SCRIPT.md`

Aim to complete the project demonstration within **5 minutes**.

### Step 9: Run the Evaluation
1. Create or locate the `evaluation/pdfs/` folder.
2. Copy the same two public legal PDF documents into this folder.
3. Run the retrieval-only evaluation:

```bash
python evaluate.py --retrieval-only
```

4. Run the evaluation with the language model:

```bash
python evaluate.py --with-llm
```

5. Check and revise the supporting phrases to match the PDF editions used.
6. Complete the manual review sheet.
7. Record actual evaluation results in:

`docs/RESULTS_TO_FILL.md`

8. Include the final evaluation metrics in the last slide of your presentation.

---

### Important Notes
- **Operating system:** These installation steps are intended for macOS.
- **Python version:** Python 3.12 is recommended for this project.
- **Local LLM:** Qwen 2.5 (3B), executed using Ollama.
- **User interface:** Streamlit.
- **Internet connection:** Required during initial installation and model downloads.
- **Offline execution:** After all required models and dependencies are downloaded, the project is designed to run locally without an internet connection.




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
