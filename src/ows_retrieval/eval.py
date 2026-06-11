"""Full-corpus BM25 evaluation with TREC retrieval metrics via ir_measures."""
import argparse
import json
from pathlib import Path

import ir_measures
import pandas as pd
from ir_measures import R, nDCG

from ows_retrieval.bm25 import build_index, doc_text, search
from ows_retrieval.data import load_docs, load_owi_docs, load_qrels, load_topics

# Narrative-stated k-values for retrieval metrics.
K_VALUES = [1, 3, 5, 10, 20, 50]

# Retrieval depth: the deepest cutoff we report metrics at.
K_RETRIEVE = K_VALUES[-1]

# Narrative-stated metric set: nDCG@k and Recall@k.
MEASURES = [nDCG @ k for k in K_VALUES] + [R @ k for k in K_VALUES]


def run_bm25_full(docs: pd.DataFrame | None = None, text_fn=doc_text) -> pd.DataFrame:
    """Index every doc, run every topic, return rankings as a DataFrame.

    Columns: topic_id, doc_id, rank, score.

    `docs` and `text_fn` select the corpus and how to read each row's text.
    They default to the original HTML index (`load_docs` + `doc_text`), so
    existing callers are unchanged. Pass the clean ir_datasets corpus
    (`docs=load_owi_docs(), text_fn=lambda row: row["default_text"]`) to build
    the index that matches the dense retriever.
    """
    topics = load_topics()
    if docs is None:
        docs = load_docs()

    retriever = build_index(docs, text_fn=text_fn)

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


# Each index variant: which corpus, how to read its text, where to write, and
# the run-file tag. "html" reproduces the original v1 run; "clean" builds the
# HTML-stripped index over default_text() that matches the dense retriever.
INDEX_VARIANTS = {
    "html": {
        "docs_fn": load_docs,
        "text_fn": doc_text,
        "experiment_dir": Path("experiments/2026-05-17_bm25_full"),
        "tag": "bm25",
    },
    "clean": {
        "docs_fn": load_owi_docs,
        "text_fn": lambda row: row["default_text"],
        "experiment_dir": Path("experiments/2026-06-09_bm25_clean_full"),
        "tag": "bm25_clean",
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a full-corpus BM25 evaluation.")
    parser.add_argument(
        "--index",
        choices=sorted(INDEX_VARIANTS),
        default="html",
        help="which document text to index: 'html' (v1) or 'clean' (matches dense)",
    )
    args = parser.parse_args()
    variant = INDEX_VARIANTS[args.index]

    experiment_dir = variant["experiment_dir"]
    experiment_dir.mkdir(parents=True, exist_ok=True)

    run_df = run_bm25_full(docs=variant["docs_fn"](), text_fn=variant["text_fn"])

    run_file = experiment_dir / f"{variant['tag']}.run.txt"
    write_trec_run(run_df, run_file, tag=variant["tag"])

    metrics = compute_metrics(run_df, load_qrels())

    for name in sorted(metrics):
        print(f"  {name:<14} {metrics[name]:.4f}")

    results_file = experiment_dir / "results.json"
    results_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
