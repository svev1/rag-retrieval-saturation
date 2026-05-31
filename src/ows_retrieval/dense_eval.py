"""Full dense (FAISS over Jina V5 nano) evaluation with TREC retrieval metrics.

Mirrors eval.py (the BM25 baseline): build the index once, run every topic,
collect rankings, then score with the same MEASURES via the shared
write_trec_run / compute_metrics helpers so the two retrievers are directly
comparable.
"""
import json
from pathlib import Path

import numpy as np

from ows_retrieval.data import (
    load_chunk_embeddings,
    load_qrels,
    load_query_embeddings,
)
from ows_retrieval.dense import build_index, normalize, search
from ows_retrieval.eval import K_RETRIEVE, compute_metrics, write_trec_run


def run_dense_full() -> "pd.DataFrame":
    """Index every chunk, run every query, return rankings as a DataFrame.

    Columns: topic_id, doc_id, rank, score.
    """
    import pandas as pd

    # Build the chunk index once (the expensive part), normalized for exact cosine.
    chunk_vectors, doc_ids = load_chunk_embeddings()
    index = build_index(normalize(chunk_vectors))

    # Query vectors; query_id equals topic_id (see data.load_query_embeddings).
    query_vectors, query_ids = load_query_embeddings()

    rows = []
    for query_vector, query_id in zip(query_vectors, query_ids):
        top_doc_ids, top_scores = search(
            index, doc_ids, query_vector, k=K_RETRIEVE
        )
        for rank, (doc_id, score) in enumerate(
            zip(top_doc_ids, top_scores), start=1
        ):
            rows.append(
                {
                    "topic_id": int(query_id),
                    "doc_id": doc_id,
                    "rank": rank,
                    "score": float(score),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    experiment_dir = Path("experiments/2026-05-31_dense_full")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    run_df = run_dense_full()

    run_file = experiment_dir / "dense.run.txt"
    write_trec_run(run_df, run_file, tag="dense")

    metrics = compute_metrics(run_df, load_qrels())

    for name in sorted(metrics):
        print(f"  {name:<14} {metrics[name]:.4f}")

    results_file = experiment_dir / "results.json"
    results_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
