"""Inspect sample-PDF retrieval locally, without calling a language model."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient
from lab_support import RagSession


QUESTIONS = (
    "How long is the return window?",
    "How long does standard delivery take?",
    "What is the international shipping policy?",
)


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return number


def collect_evidence(questions, *, k=2, chunk_size=500, overlap=80):
    """Return real passages; do not generate answers or infer support from scores."""
    if not questions or any(not question.strip() for question in questions):
        raise ValueError("Provide at least one nonempty question.")
    if k < 1 or chunk_size < 1 or not 0 <= overlap < chunk_size:
        raise ValueError("Require k >= 1, chunk_size > 0, and 0 <= overlap < chunk_size.")

    # Explicit local storage prevents a saved QDRANT_URL from uploading the PDF.
    qdrant = QdrantClient(":memory:")
    try:
        rag = RagSession(qdrant=qdrant)
        count = rag.ingest(
            [ROOT / "data/sample_handbook.pdf"],
            chunk_size=chunk_size, overlap=overlap,
        )
        cases = []
        for question in questions:
            hits = rag.retrieve(question, k=k)
            cases.append({
                "question": question,
                "passages": [
                    {"rank": rank, "source": hit.payload["source"],
                     "page": hit.payload["page"], "text": hit.payload["text"],
                     "similarity": round(float(hit.score), 6)}
                    for rank, hit in enumerate(hits, 1)
                ],
            })
        return {
            "mode": "retrieval_only", "model_calls": 0,
            "storage": "local_memory", "chunks": count,
            "settings": {"k": k, "chunk_size": chunk_size, "overlap": overlap},
            "cases": cases,
        }
    finally:
        qdrant.close()


def render_text(report):
    lines = [
        "A search result is not an answer",
        f"Indexed {report['chunks']} chunks. Local retrieval only; no model calls.",
        "Read the passages before deciding whether an answer is supported.",
    ]
    for case in report["cases"]:
        lines.extend(["", f"Question: {case['question']}"])
        for passage in case["passages"]:
            lines.extend([
                f"  [{passage['rank']}] {passage['source']}, page {passage['page']}"
                f" | cosine similarity {passage['similarity']:.3f}",
                "  " + " ".join(passage["text"].split()),
            ])
    lines.extend(["", "Similarity ranks passages; it is not a confidence or support verdict."])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", help="Try one custom question instead of the three examples.")
    parser.add_argument("--k", type=positive_integer, default=2)
    parser.add_argument("--chunk-size", type=positive_integer, default=500)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--json", action="store_true", help="Print evidence as JSON.")
    args = parser.parse_args(argv)
    if not 0 <= args.overlap < args.chunk_size:
        parser.error("Require 0 <= overlap < chunk-size.")
    if args.question is not None and not args.question.strip():
        parser.error("Question must contain text.")
    questions = (args.question,) if args.question is not None else QUESTIONS
    report = collect_evidence(
        questions, k=args.k, chunk_size=args.chunk_size, overlap=args.overlap,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True) if args.json else render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
