"""Quick sanity check on the V5 embeddings before we wire them in.

For 5 topics with relevant docs in qrels:
  - Look up the topic's query vector in radboud-val.jina-v5-nano.queries.parquet.
  - For up to 3 highly-relevant docs, fetch all their chunks from
    radboud-val.jina-v5-nano.chunks-1024.parquet and compute cosine
    against the query (max across chunks).
  - For 3 random unjudged docs, do the same.
  - Report relevant vs random cosines per topic + overall.

Both sides are pre-normalized in the parquets, so cosine = inner product.
Run from the repo root. Writes a UTF-8 log to v5_sanity.out next to itself.
"""
import os
import numpy as np
import pandas as pd
from ows_retrieval.data import load_qrels, load_topics

OUT_PATH = os.path.join(os.path.dirname(__file__) or '.', 'v5_sanity.out')
out_file = open(OUT_PATH, 'w', encoding='utf-8')


def log(msg=''):
    print(msg)
    out_file.write(msg + '\n')
    out_file.flush()


CHUNKS = 'data/raw/radboud-val.jina-v5-nano.chunks-1024.parquet'
QUERIES = 'data/raw/radboud-val.jina-v5-nano.queries.parquet'

log("Loading V5 query embeddings...")
qdf = pd.read_parquet(QUERIES, columns=['query_id', 'embedding'])
log(f"  queries loaded: {len(qdf)}  (each is {len(qdf.iloc[0]['embedding'])}-d)")
# Index by query_id (string) for fast lookup.
qdf['query_id'] = qdf['query_id'].astype(str)
q_by_id = {r['query_id']: np.asarray(r['embedding'], dtype=np.float32) for _, r in qdf.iterrows()}

log("Loading V5 chunk embeddings (this is the big one, ~300k rows)...")
cdf = pd.read_parquet(CHUNKS, columns=['doc_id', 'embedding'])
log(f"  chunks loaded: {len(cdf)}  (each is {len(cdf.iloc[0]['embedding'])}-d)")
log(f"  distinct docs: {cdf['doc_id'].nunique()}")

# Build a doc_id -> stacked chunk matrix for fast lookup.
log("Grouping chunks by doc_id...")
grouped = {}
for doc_id, sub in cdf.groupby('doc_id'):
    grouped[doc_id] = np.stack([np.asarray(v, dtype=np.float32) for v in sub['embedding']])
log(f"  grouped: {len(grouped)} docs")
log()

topics = load_topics()
qrels = load_qrels()

# Pick 5 topics with relevant docs.
sample_tids = qrels[qrels['rel'] > 0]['topic_id'].drop_duplicates().head(5).tolist()
rng = np.random.default_rng(seed=42)
all_doc_ids = list(grouped.keys())


def max_cos_with_doc(q_vec, doc_id):
    """Max cosine over a doc's chunks. Returns None if doc not in the chunk index."""
    mat = grouped.get(doc_id)
    if mat is None:
        return None
    return float((mat @ q_vec).max())


all_rel = []
all_rand = []

for tid in sample_tids:
    qvec = q_by_id.get(str(tid))
    if qvec is None:
        log(f"Topic {tid}: NO QUERY EMBEDDING FOUND (query_id {tid!r} not in queries parquet)")
        continue
    query_text = topics[topics['topic_id'] == tid].iloc[0]['query']

    rel = qrels[(qrels['topic_id'] == tid) & (qrels['rel'] > 0)]
    rel = rel.sort_values('rel', ascending=False).head(3)
    judged = set(qrels[qrels['topic_id'] == tid]['doc_id'])
    random_pool = [d for d in all_doc_ids if d not in judged]
    rand_ids = rng.choice(random_pool, size=3, replace=False).tolist()

    log(f"Topic {tid}: {query_text!r}")
    log(f"  {'kind':10s}  {'rel':>3s}  {'cos':>7s}  doc_id")

    for _, qr in rel.iterrows():
        d, g = qr['doc_id'], int(qr['rel'])
        c = max_cos_with_doc(qvec, d)
        if c is None:
            log(f"  {'relevant':10s}  {g:>3d}  {'(none)':>7s}  {d[:16]}..  (not in chunks parquet)")
        else:
            log(f"  {'relevant':10s}  {g:>3d}  {c:7.4f}  {d[:16]}..")
            all_rel.append(c)
    for d in rand_ids:
        c = max_cos_with_doc(qvec, d)
        if c is None:
            log(f"  {'random':10s}  {'-':>3s}  {'(none)':>7s}  {d[:16]}..")
        else:
            log(f"  {'random':10s}  {'-':>3s}  {c:7.4f}  {d[:16]}..")
            all_rand.append(c)
    log()

if all_rel and all_rand:
    log(f"=== overall ===")
    log(f"  relevant : n={len(all_rel)}  mean={np.mean(all_rel):.4f}  min={np.min(all_rel):.4f}  max={np.max(all_rel):.4f}")
    log(f"  random   : n={len(all_rand)}  mean={np.mean(all_rand):.4f}  min={np.min(all_rand):.4f}  max={np.max(all_rand):.4f}")
    log(f"  gap (rel mean - rand mean) = {np.mean(all_rel) - np.mean(all_rand):+.4f}")
    log()
    log("Pass criterion: gap clearly positive (e.g. > +0.2), and every")
    log("relevant cosine is higher than the random max.")

out_file.close()
print(f"\nlog written to {OUT_PATH}")
