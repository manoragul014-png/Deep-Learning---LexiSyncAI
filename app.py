"""LexiSyncAI Streamlit application. Run: python -m streamlit run app.py"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import fitz
import streamlit as st
from PIL import Image, ImageDraw

from lexisync.ingest import load_or_extract
from lexisync.retrieval import Retriever
from lexisync.generation import create_answer, has_reported_conflict

BASE = Path(__file__).resolve().parent
CACHE = BASE / "data" / "cache"
CASE_DIR = BASE / "data" / "cases"

st.set_page_config(page_title="LexiSyncAI | Legal RAG", page_icon="⚖️", layout="wide")
st.title("⚖️ LexiSyncAI")
st.caption("Evidence-grounded legal document analyst | Document → Search → Rerank → Answer → Verify")


def safe_name(s):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", Path(s).name)[:100]


def picture(pdf_path: Path, page: int, bbox=None):
    doc = fitz.open(str(pdf_path))
    pg = doc[page-1]
    scale = 1.25
    pix = pg.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    if bbox and len(bbox) == 4:
        draw = ImageDraw.Draw(image, "RGBA")
        xy = [v * scale for v in bbox]
        draw.rectangle(xy, fill=(255, 205, 20, 45), outline=(217, 110, 0, 225), width=3)
    doc.close()
    return image


with st.sidebar:
    st.header("Settings")
    exam = st.toggle("Unseen-PDF exam mode (allow 1 PDF)", value=False,
                     help="Normal assignment: 2–5 public PDFs. The oral exam may have only 1 unseen PDF.")
    tables = st.checkbox("Extract tables (best effort)", value=True)
    provider = st.radio("LLM provider", ["Ollama", "OpenAI API"], horizontal=False)
    if provider == "Ollama":
        model = st.text_input("Local Ollama model", "qwen2.5:3b")
        api_key = None
        st.caption("One-time: `ollama pull qwen2.5:3b`. Start Ollama before the demo.")
    else:
        model = st.text_input("API model", "gpt-4o-mini")
        api_key = st.text_input("API key (not saved)", type="password")
        st.caption("API use requires separate billing; only upload public documents.")
    st.divider()
    st.caption("Important: This is an educational prototype, not legal advice. Never upload private client material.")

st.subheader("1. Create a case and upload legal PDFs")
case_name = st.text_input("Case name", "EU Data Protection and AI Act")
files = st.file_uploader("Upload public, digital-text legal PDFs", type=["pdf"],
                         accept_multiple_files=True, help="2–5 PDFs for the assignment.")
min_files = 1 if exam else 2
valid_count = files is not None and min_files <= len(files) <= 5
if files and not valid_count:
    st.warning(f"Upload {min_files}–5 public PDFs before indexing.")

if st.button("Index documents / pre-warm models", type="primary", disabled=not valid_count):
    try:
        blobs = [(safe_name(f.name), f.getvalue()) for f in files]
        if len({name for name, _ in blobs}) != len(blobs):
            raise ValueError("Two files have the same name; please rename one.")
        case_hash = hashlib.sha256(b"".join(hashlib.sha256(b).digest() for _, b in blobs)).hexdigest()[:14]
        slug = re.sub(r"[^a-z0-9-]+", "-", case_name.lower()).strip("-")[:45] or "case"
        folder = CASE_DIR / f"{slug}_{case_hash}"
        folder.mkdir(parents=True, exist_ok=True)
        all_chunks, stats, paths = [], [], {}
        with st.spinner("Parsing PDF pages, sections, and optional tables…"):
            for name, data in blobs:
                p = folder / name
                p.write_bytes(data)
                paths[name] = p
                chunks, detail = load_or_extract(data, name, CACHE, include_tables=tables)
                all_chunks.extend(chunks)
                stats.append(detail)
        if not all_chunks:
            raise ValueError("No extractable text. Image-only PDFs need OCR, which this version does not implement.")
        with st.spinner("Loading local embedding and reranker models; embedding passages…"):
            retriever = Retriever(all_chunks, CACHE).build()
        st.session_state["case"] = {"name": case_name, "stats": stats,
                                      "paths": paths, "chunks": all_chunks,
                                      "retriever": retriever, "case_hash": case_hash}
        st.session_state.pop("answer", None)
        st.session_state.pop("evidence", None)
        st.success(f"Indexed {len(stats)} document(s) and {len(all_chunks)} passages. "
                   f"Models warmed in {retriever.index_seconds:.1f}s.")
    except Exception as exc:
        st.error(f"Indexing failed: {exc}")
        st.info("Check the installation instructions in README.md. Do not claim a successful demo until the models load.")

case = st.session_state.get("case")
if not case:
    st.info("Upload your public PDFs and click **Index documents / pre-warm models**. "
            "The first run downloads model weights; indexing later runs can reuse cached vectors.")
    st.stop()

st.subheader("2. Document structure and quality")
rows = [{"Document": x.document, "PDF pages": x.pages, "Passages": x.chunks,
         "Sections": x.sections_detected, "Tables": x.tables_detected,
         "Image-only page flags": ", ".join(map(str, x.scanned_pages)) or "None"}
        for x in case["stats"]]
st.dataframe(rows, use_container_width=True, hide_index=True)
if any(d.scanned_pages for d in case["stats"]):
    st.warning("Some pages look scanned. They are flagged; OCR is not implemented. "
               "Questions depending on their content may need abstention.")
with st.expander("What was implemented in ingestion?"):
    st.write("Position-aware PDF text blocks; heuristic multi-column reading order; "
             "repeated running header/footer removal; physical PDF page numbers; "
             "article/section-aware chunks; best-effort table extraction; image-only flags; "
             "cached embeddings. Region highlights indicate extracted block boxes and can be approximate.")

st.subheader("3. Ask a question with verifiable evidence")
question = st.text_input("Legal question", placeholder="Under Article 83, what are the two GDPR fine tiers?")
mode = st.selectbox("Retrieval configuration", ["hybrid_rerank", "hybrid", "dense", "bm25"],
                    format_func=lambda x: {
                        "hybrid_rerank": "BM25 + dense + RRF + cross-encoder (final)",
                        "hybrid": "BM25 + dense + RRF (ablation)",
                        "dense": "Dense embeddings only (ablation)",
                        "bm25": "BM25 only (ablation)"}[x])
num_results = st.slider("Evidence passages", min_value=3, max_value=12, value=8)
if st.button("Answer and show sources", disabled=not question.strip(), type="primary"):
    try:
        with st.spinner("Retrieving relevant passages and generating a grounded answer…"):
            matches, timings = case["retriever"].rank(question, mode=mode, top_k=num_results)
            result = create_answer(question, matches, provider=provider, model=model,
                                   api_key=api_key, timings=timings,
                                   all_documents=set(case["paths"]))
        st.session_state["answer"] = result
        st.session_state["evidence"] = matches
    except Exception as exc:
        st.error(f"Could not generate an answer: {exc}")
        st.info("If Ollama is offline, start it and pull the chosen model. "
                "Do not replace a failed LLM answer with an uncited guess.")

answer = st.session_state.get("answer")
evidence = st.session_state.get("evidence")
if answer and evidence:
    left, right = st.columns([1, 1], gap="large")
    with left:
        if not answer["answerable"]:
            st.error("INSUFFICIENT EVIDENCE — Abstained")
            st.write(answer["explanation"])
            st.write("**Reason:** " + answer["missing_evidence"])
        else:
            message = "Evidence check passed | " + answer["confidence"]
            if answer["confidence"] == "LOW":
                st.warning(message)
            else:
                st.success(message)
            st.caption("Confidence is a rule-based citation-coverage/conflict flag, not a probability. "
                       "LOW means the LLM reported a possible conflict; it does not prove a contradiction.")
            for claim in answer["claims"]:
                ids = ", ".join(f"[{x['id']}]" for x in claim["citations"])
                st.markdown(f"- {claim['statement']} **{ids}**")
            st.markdown("**Why the evidence supports the answer**")
            st.write(answer["explanation"])
            if has_reported_conflict(answer.get("conflicts")):
                st.warning("Potential conflict: " + answer["conflicts"])
            with st.expander("Confidence diagnostic (model conflict response)"):
                st.code(repr(answer.get("conflicts", "")), language=None)
                st.caption("If LOW appears without a real contradiction, check this field. "
                           "The label is rule-based and does not measure factual correctness.")
        st.caption("Documents searched: " + ", ".join(answer["documents_searched"]))
        st.markdown("**Timing / token evidence**")
        st.json({**answer.get("timings", {}), **answer.get("usage", {})})
        st.download_button("Download answer and evidence (JSON)",
                           data=json.dumps({"answer": answer,
                                            "evidence": [{"chunk": row["chunk"].as_dict(),
                                                          "score": row["score"]} for row in evidence]},
                                           ensure_ascii=False, indent=2),
                           file_name="lexisync_result.json", mime="application/json")
    with right:
        st.markdown("**Retrieved evidence and cited PDF page**")
        choices = list(range(len(evidence)))
        chosen = st.selectbox("Choose evidence passage", choices,
                              format_func=lambda i: f"E{i+1} — {evidence[i]['chunk'].document} · "
                                  f"p. {evidence[i]['chunk'].page} · {evidence[i]['chunk'].section}")
        c = evidence[chosen]["chunk"]
        st.caption(f"Source E{chosen+1}: {c.document}, physical PDF p. {c.page}, "
                   f"{c.section} | {c.kind}")
        st.code(c.text[:4000], language=None)
        pth = case["paths"].get(c.document)
        if pth and pth.exists():
            st.image(picture(pth, c.page, c.bbox),
                     caption="Original PDF page; highlighted extracted region (approximate).",
                     use_container_width=True)
        if answer["citations"]:
            with st.expander("Validated claim-level quotations"):
                for cite in answer["citations"]:
                    st.write(f"[{cite['id']}] {cite['document']} · p. {cite['page']} · {cite['section']}")
                    st.code(cite["quote"])

st.divider()
st.caption("RAG limits: extraction quality and retrieval recall may affect accuracy. "
           "A matched quotation alone does not prove semantic entailment; verify legal interpretations manually.")
