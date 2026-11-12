"""Run reproducible retrieval comparison and optional LLM evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from lexisync.ingest import load_or_extract
from lexisync.retrieval import Retriever
from lexisync.generation import create_answer

BASE = Path(__file__).resolve().parent


def normalized(s: str):
    return " ".join(s.lower().split())


def gold_chunk_indices(chunks, phrases):
    """One relevant chunk for each manually validated gold support phrase."""
    indices = []
    for phrase in phrases:
        hits = {i for i, c in enumerate(chunks)
                if normalized(phrase) in normalized(c.text)}
        indices.append(hits)
    return indices


def coverage_at_k(returned, label_groups, k):
    chosen = {r["index"] for r in returned[:k]}
    return float(all(chosen.intersection(g) for g in label_groups))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", type=Path, default=BASE / "evaluation" / "pdfs")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("--provider", default="Ollama", choices=["Ollama", "OpenAI API"])
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--out", type=Path, default=BASE / "evaluation" / "results")
    args = ap.parse_args()
    if not args.retrieval_only and not args.with_llm:
        ap.error("Choose --retrieval-only or --with-llm")
    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if len(pdfs) < 2 or len(pdfs) > 5:
        ap.error("Put 2–5 public PDF files in evaluation/pdfs (or use --pdf-dir PATH).")
    chunks, stats = [], []
    for pdf in pdfs:
        c, s = load_or_extract(pdf.read_bytes(), pdf.name, BASE / "data" / "cache")
        chunks.extend(c)
        stats.append(vars(s))
    retriever = Retriever(chunks, BASE / "data" / "cache").build()
    questions = json.loads((BASE / "evaluation" / "questions.json").read_text())
    rows, review = [], []
    for q in questions:
        record = {"id": q["id"], "type": q["type"], "question": q["question"],
                  "expected_answer": q["expected_answer"],
                  "answerable_gold": q["answerable"]}
        gold = gold_chunk_indices(chunks, q["support_phrases"])
        label_valid = bool(q["answerable"] and gold and all(gold))
        record["gold_phrases_located"] = label_valid
        for mode in ["bm25", "dense", "hybrid", "hybrid_rerank"]:
            top, times = retriever.rank(q["question"], mode=mode, top_k=10)
            record[f"{mode}_latency_s"] = round(times["retrieval_seconds"] +
                                                 times["reranking_seconds"], 3)
            record[f"{mode}_top_ids"] = [r["chunk"].id for r in top]
            # Exclude missing/unverified phrase labels from the recall denominator.
            if label_valid:
                for k in [3, 5, 10]:
                    record[f"{mode}_recall@{k}"] = coverage_at_k(top, gold, k)
            if mode == "hybrid_rerank" and args.with_llm:
                output = create_answer(q["question"], top, provider=args.provider,
                                       model=args.model, api_key=args.api_key,
                                       timings=times, all_documents={s["document"] for s in stats})
                record["predicted_answerable"] = output["answerable"]
                record["generated_claims"] = [c["statement"] for c in output["claims"]]
                record["citations"] = output["citations"]
                record["generation_latency_s"] = output["timings"].get("generation_seconds")
                record["input_tokens"] = output["usage"].get("input_tokens")
                record["output_tokens"] = output["usage"].get("output_tokens")
                # Manual adjudication is mandatory: substring presence != entailment.
                review.append({"id": q["id"], "question": q["question"],
                               "expected_answer": q["expected_answer"],
                               "generated_claims": json.dumps(record["generated_claims"], ensure_ascii=False),
                               "cited_sources": json.dumps(output["citations"], ensure_ascii=False),
                               "answer_correct_manual_0_or_1": "",
                               "citation_precision_manual_0_to_1": "",
                               "reviewer_notes": ""})
        rows.append(record)
    summary = {"n_documents": len(stats), "n_passages": len(chunks),
               "n_questions": len(questions), "n_answerable_gold": sum(q["answerable"] for q in questions),
               "n_unanswerable_gold": sum(not q["answerable"] for q in questions),
               "n_verified_recall_labels": sum(r["gold_phrases_located"] for r in rows),
               "missing_label_ids": [r["id"] for r in rows
                                     if r["answerable_gold"] and not r["gold_phrases_located"]],
               "index_seconds": retriever.index_seconds,
               "documents": stats, "recall": {}, "latency": {}}
    for mode in ["bm25", "dense", "hybrid", "hybrid_rerank"]:
        for k in [3, 5, 10]:
            key = f"{mode}_recall@{k}"
            values = [r[key] for r in rows if key in r]
            summary["recall"][key] = round(statistics.mean(values), 3) if values else None
        delays = [r[f"{mode}_latency_s"] for r in rows]
        summary["latency"][f"{mode}_median_s"] = round(statistics.median(delays), 3)
    if args.with_llm:
        answerable_gold = [r for r in rows if r["answerable_gold"]]
        unanswerable_gold = [r for r in rows if not r["answerable_gold"]]
        summary["abstention_accuracy_on_unanswerable"] = round(
            statistics.mean(not r["predicted_answerable"] for r in unanswerable_gold), 3)
        summary["false_abstention_rate_on_answerable"] = round(
            statistics.mean(not r["predicted_answerable"] for r in answerable_gold), 3)
        summary["manual_correctness"] = "PENDING REVIEW CSV"
        summary["manual_citation_precision"] = "PENDING REVIEW CSV"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.with_suffix(".json")).write_text(
        json.dumps({"summary": summary, "per_question": rows}, ensure_ascii=False, indent=2))
    if review:
        with (args.out.with_name(args.out.name + "_manual_review.csv")).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(review[0]))
            writer.writeheader()
            writer.writerows(review)
    print(json.dumps(summary, indent=2))
    if summary["missing_label_ids"]:
        print("WARNING: gold text not found; review PDF edition/phrase labels. "
              "Reported recall excludes these items and is NOT a complete 15-question recall result.")
    print("Results written to", args.out.with_suffix(".json"))


if __name__ == "__main__":
    main()
