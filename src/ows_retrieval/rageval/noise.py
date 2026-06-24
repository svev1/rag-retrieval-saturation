"""Measure retrieval noise per (retriever, topic, k) from the qrels.

For each retriever's top-k, every retrieved document falls in one of three buckets
by its qrels label: relevant (label >= 1), judged non-relevant (label <= 0, i.e.
the 0s and the -2 junk), or unjudged (absent from qrels -- nobody assessed it).

Noise is the share of the top-k that is not relevant. Unjudged docs grow with k
(deep ranks were never pooled), so we report two versions and break unjudged out
on its own rather than hiding the choice:
  - ``noise``        = (non-relevant + unjudged) / k   -- inclusive
  - ``noise_strict`` = non-relevant / k                -- only confirmed junk

No answers or LLM are involved: this depends only on the run files and qrels, so
it is unaffected by which generator produced the answers.
"""
import argparse
from pathlib import Path

import pandas as pd

from ows_retrieval.data import load_qrels
from ows_retrieval.rag.context import RUN_FILES, ranked_docs_by_topic

DEFAULT_RETRIEVERS = ["bm25_clean", "dense"]
DEFAULT_K_VALUES = [1, 3, 5, 10, 20, 50]
REL_THRESHOLD = 1  # qrels label >= 1 counts as relevant; 0 and -2 do not


def compute_noise(
    retrievers: list[str] = DEFAULT_RETRIEVERS,
    k_values: list[int] = DEFAULT_K_VALUES,
) -> pd.DataFrame:
    """One row per (retriever, topic, k) with the three buckets and noise rates."""
    # {(topic_id, doc_id): label} for O(1) per-document lookup, from the shared loader.
    qrels_df = load_qrels()
    qrels = dict(zip(zip(qrels_df["topic_id"], qrels_df["doc_id"]), qrels_df["rel"]))
    rows = []
    for retriever in retrievers:
        ranked = ranked_docs_by_topic(RUN_FILES[retriever])
        for topic_id, doc_ids in ranked.items():
            for k in k_values:
                topk = doc_ids[:k]
                labels = [qrels.get((topic_id, d)) for d in topk]
                n_relevant = sum(1 for x in labels if x is not None and x >= REL_THRESHOLD)
                n_unjudged = sum(1 for x in labels if x is None)
                n_nonrelevant = len(topk) - n_relevant - n_unjudged
                n = len(topk)  # = k unless the run held fewer than k docs for a topic
                rows.append(
                    {
                        "retriever": retriever,
                        "topic_id": topic_id,
                        "k": k,
                        "n_relevant": n_relevant,
                        "n_nonrelevant": n_nonrelevant,
                        "n_unjudged": n_unjudged,
                        "noise": (n_nonrelevant + n_unjudged) / n,
                        "noise_strict": n_nonrelevant / n,
                        "unjudged_rate": n_unjudged / n,
                        "relevant_rate": n_relevant / n,
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values(["retriever", "topic_id", "k"])
        .reset_index(drop=True)
    )


def main() -> None:
    """CLI entry point: write the noise table, print mean noise by k."""
    parser = argparse.ArgumentParser(description="Measure retrieval noise from qrels.")
    parser.add_argument("out", type=Path, help="output .parquet path")
    args = parser.parse_args()

    df = compute_noise()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)

    print(f"{len(df)} rows | {df['topic_id'].nunique()} topics | retrievers {sorted(df['retriever'].unique())}")
    print("\nmean noise (inclusive) by retriever x k:")
    print(df.pivot_table(index="retriever", columns="k", values="noise").round(2))
    print("\nmean unjudged_rate by retriever x k:")
    print(df.pivot_table(index="retriever", columns="k", values="unjudged_rate").round(2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
