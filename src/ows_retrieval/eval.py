"""Full-corpus BM25 evaluation with TREC retrieval metrics via ir_measures."""
import json
from pathlib import Path

import ir_measures
import pandas as pd
from ir_measures import R, nDCG

from ows_retrieval.bm25 import build_index, search
from ows_retrieval.data import load_docs, load_qrels, load_topics

# Narrative-stated k-values for retrieval metrics.
K_VALUES = [1, 3, 5, 10, 20, 50]

# Retrieval depth: the deepest cutoff we report metrics at.
K_RETRIEVE = K_VALUES[-1]

# Narrative-stated metric set: nDCG@k and Recall@k.
MEASURES = [nDCG @ k for k in K_VALUES] + [R @ k for k in K_VALUES]


def run_bm25_full() -> pd.DataFrame:
    """Index every doc, run every topic, return rankings as a DataFrame.

    Columns: topic_id, doc_id, rank, score.
    """
    topics = load_topics()
    docs = load_docs()

    retriever = build_index(docs)

    rows = []
    for _, t in topics.iterrows():
        positions, scores = search(retriever, t["query"], k=K_RETRIEVE)
        for rank, (pos, score) in enumerate(zip(positions, scores), start=1):
            rows.append(
                {
                    "topic_id": int(t["topic_id"]),
                    "doc_id": docs.iloc[pos]["id"],
                    "rank": rank,
                    "score": float(score),
                }
            )
    return pd.DataFrame(rows)


def write_trec_run(run_df: pd.DataFrame, path: Path, tag: str = "bm25") -> None:
    """Write a run DataFrame to TREC run-file format.

    Each line: topic_id Q0 doc_id rank score tag
    """
    lines = [
        f"{row['topic_id']} Q0 {row['doc_id']} {row['rank']} {row['score']:.6f} {tag}"
        for _, row in run_df.iterrows()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_metrics(run_df: pd.DataFrame, qrels_df: pd.DataFrame) -> dict:
    """Compute MEASURES via ir_measures.calc_aggregate.

    Negative qrels values (spam) are clipped to 0.
    """
    qrels_for_eval = pd.DataFrame(
        {
            "query_id": qrels_df["topic_id"].astype(str),
            "doc_id": qrels_df["doc_id"].astype(str),
            "relevance": qrels_df["rel"].clip(lower=0).astype(int),
        }
    )
    run_for_eval = pd.DataFrame(
        {
            "query_id": run_df["topic_id"].astype(str),
            "doc_id": run_df["doc_id"].astype(str),
            "score": run_df["score"].astype(float),
        }
    )
    raw = ir_measures.calc_aggregate(MEASURES, qrels_for_eval, run_for_eval)
    return {str(measure): float(value) for measure, value in raw.items()}


if __name__ == "__main__":
    experiment_dir = Path("experiments/2026-05-17_bm25_full")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    run_df = run_bm25_full()

    run_file = experiment_dir / "bm25.run.txt"
    write_trec_run(run_df, run_file)

    metrics = compute_metrics(run_df, load_qrels())

    for name in sorted(metrics):
        print(f"  {name:<14} {metrics[name]:.4f}")

    results_file = experiment_dir / "results.json"
    results_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
