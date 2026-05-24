"""Hand-eyeball inspection of retrieval results on the smoke slice.

Provides two entry points:
  - smoke_bm25(query_ids):          print top-k tables to the terminal.
  - smoke_experiment(query_ids):    save the same table as results.json in 
                                    experiments/<dat>_bm25_smoke/.
"""
import json
from datetime import date
from pathlib import Path

import pandas as pd

from ows_retrieval.bm25 import build_index, search
from ows_retrieval.sample import build_smoke_slice

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _smoke_rows(topics, qrels, docs, retriever, k):
    """Internal helper: return {topic_id: [row dicts]} for the smoke run."""
    output = {}
    for _, topic in topics.iterrows():
        positions, scores = search(retriever, topic["query"], k=k)
        rows = []
        for rank, (pos, score) in enumerate(zip(positions, scores), start=1):
            doc = docs.iloc[pos]
            judged = qrels[
                (qrels["topic_id"] == topic["topic_id"])
                & (qrels["doc_id"] == doc["id"])
            ]
            rel = int(judged["rel"].iloc[0]) if len(judged) else None
            rows.append(
                {
                    "rank": rank,
                    "score": round(float(score), 4),
                    "rel": rel,  # None means unjudged
                    "doc_id": doc["id"],
                    "title": (doc["title"] if pd.notna(doc["title"]) else None),
                }
            )
        output[int(topic["topic_id"])] = {"query": topic["query"], "results": rows}
    return output


def smoke_bm25(
    query_ids, k: int = 10, n_distractors: int = 500, seed: int = 42
) -> None:
    """Run BM25 on the smoke slice and print top-k per query with relevance labels."""
    topics, qrels, docs = build_smoke_slice(
        query_ids, n_distractors=n_distractors, seed=seed
    )
    retriever = build_index(docs)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)

    output = _smoke_rows(topics, qrels, docs, retriever, k)
    for topic_id, content in output.items():
        rows = [
            {
                "rank": r["rank"],
                "score": r["score"],
                "rel": "-" if r["rel"] is None else r["rel"],
                "title": (r["title"] or "(no title)")[:80],
            }
            for r in content["results"]
        ]
        print(f"\n=== topic {topic_id}: {content['query']!r} ===")
        print(pd.DataFrame(rows).to_string(index=False))


def smoke_experiment(
    query_ids, k: int = 10, n_distractors: int = 500, seed: int = 42
) -> Path:
    """Run BM25 smoke and write the experiment folder.

    Creates experiments/<date>_bm25_smoke/results.json with one entry per query
    (rank, score, rel, doc_id, title).
    """
    topics, qrels, docs = build_smoke_slice(
        query_ids, n_distractors=n_distractors, seed=seed
    )
    retriever = build_index(docs)
    output = _smoke_rows(topics, qrels, docs, retriever, k)

    experiment_dir = (
        PROJECT_ROOT / "experiments" / f"{date.today().isoformat()}_bm25_smoke"
    )    
    experiment_dir.mkdir(parents=True, exist_ok=True)

    results_file = experiment_dir / "results.json"
    results_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return experiment_dir


if __name__ == "__main__":
    smoke_bm25([24, 31, 32, 52])
