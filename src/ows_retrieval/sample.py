"""Build a small smoke-test slice of the corpus for hand-eyeballing retrieval."""
import numpy as np
import pandas as pd

from .data import load_docs, load_qrels, load_topics


def build_smoke_slice(query_ids, n_distractors=500, seed=42):
    """Return a small (topics, qrels, docs) trio for the given queries.

    The smoke corpus is the union of:
      - every document that has at least one qrels row for query_ids
        (the "judged" set, around 150-200 docs for 4 queries)
      - n_distractors random documents drawn from outside the judged set 
    """
    topics = load_topics()
    qrels = load_qrels()
    docs = load_docs()

    # 1. Filter topics and qrels to just the chosen queries.
    topics = topics[topics["topic_id"].isin(query_ids)].reset_index(drop=True)
    qrels = qrels[qrels["topic_id"].isin(query_ids)].reset_index(drop=True)

    # 2. The judged set: every doc that has any qrels row for our queries.
    judged_ids = set(qrels["doc_id"])
    judged_docs = docs[docs["id"].isin(judged_ids)]

    # 3. Distractors: random docs that are NOT judged.
    non_judged = docs[~docs["id"].isin(judged_ids)]
    rng = np.random.default_rng(seed)
    distractor_pos = rng.choice(len(non_judged), size=n_distractors, replace=False)
    distractors = non_judged.iloc[distractor_pos]

    # 4. Smoke corpus: judged + distractors, with a clean integer index.
    smoke_docs = pd.concat([judged_docs, distractors], ignore_index=True)

    return topics, qrels, smoke_docs
