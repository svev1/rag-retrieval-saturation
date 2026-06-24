"""Collate the cached RAG answers into one table for evaluation.

The sweep saved one JSON per (retriever, topic, k) answer, nested as
``t<topic>/<retriever>__k<k>.json``. This gathers them into a single table and
turns each record's ``doc_ids`` into the context text the generator saw (the
cache stores ids only). Text is shaped exactly as in generation
(``rag.context.shape_document``) but emitted as a list -- one string per document,
without the ``[Document N]`` labels -- which is the shape the scorer expects.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from ows_retrieval.rag.context import DEFAULT_CHAR_CAP, load_doc_text, shape_document

# The string the prompt tells the generator to emit when the context is
# insufficient. Flagged because the scorer handles these specially (answer
# relevancy ~0; faithfulness undefined when there are no claims to verify).
REFUSAL = "The documents do not answer this."


def collate_cache(
    cache_dir: Path,
    doc_text_source: str = "clean",
    char_cap: int = DEFAULT_CHAR_CAP,
) -> pd.DataFrame:
    """Gather every cached answer under ``cache_dir`` into one DataFrame.

    One row per (retriever, topic, k); ``contexts`` is a list of shaped documents
    in rank order. ``doc_text_source`` and ``char_cap`` must match the sweep that
    wrote the cache (the clean-index sweep used "clean" / 500).
    """
    text_map = load_doc_text(doc_text_source)
    rows = []
    for path in sorted(Path(cache_dir).glob("t*/*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "retriever": rec["retriever"],
                "topic_id": rec["topic_id"],
                "k": rec["k"],
                "question": rec["query"],
                "answer": rec["answer"],
                "contexts": [shape_document(text_map[d], char_cap) for d in rec["doc_ids"]],
                "doc_ids": rec["doc_ids"],
                "n_contexts": len(rec["doc_ids"]),
                "is_refusal": rec["answer"].strip() == REFUSAL,
                "model": rec["model"],
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["retriever", "topic_id", "k"])
        .reset_index(drop=True)
    )


def main() -> None:
    """CLI entry point: read a cache dir, write the table, print a sanity check."""
    parser = argparse.ArgumentParser(description="Collate cached RAG answers into one table.")
    parser.add_argument("cache_dir", type=Path, help="cache dir (contains t<topic>/*.json)")
    parser.add_argument("out", type=Path, help="output .parquet path")
    parser.add_argument("--doc-text-source", default="clean", help="text source the sweep used (clean|html)")
    parser.add_argument("--char-cap", type=int, default=DEFAULT_CHAR_CAP)
    args = parser.parse_args()

    df = collate_cache(args.cache_dir, args.doc_text_source, args.char_cap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)

    print(
        f"{len(df)} rows | {df['topic_id'].nunique()} topics | "
        f"retrievers {sorted(df['retriever'].unique())} | "
        f"k {sorted(int(x) for x in df['k'].unique())}"
    )
    print(f"refusals: {df['is_refusal'].sum()} | empty answers: {(df['answer'].str.strip() == '').sum()}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
