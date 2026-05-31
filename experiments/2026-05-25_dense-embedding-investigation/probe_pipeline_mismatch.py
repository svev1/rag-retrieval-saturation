"""Probe whether any Jina V3 task mode reproduces the stored doc vectors,
and whether high cosines (if any) actually mean "same doc" or just "any
two no-task vectors look alike."

Two tests:

  A. Per-doc, per-task cosine: for each of ~10 single-chunk docs, slice
     its chunk text and re-encode it with each Jina V3 task mode. Compare
     cosine against the stored vector for that doc. If a task is the one
     the upstream used, we expect ~0.95.

  B. Cross-doc baseline in the best task mode found in (A): for each
     re-encoded doc, also compute cosine against the stored vectors of
     the OTHER docs. If those off-diagonal cosines are far below the
     on-diagonal ones, the high-cosine match in (A) is meaningful. If
     they are similar, the task is non-discriminative and the match
     was an illusion.

Writes a UTF-8 log next to itself (probe_pipeline_mismatch.out)."""
import os
import sys
import numpy as np
import pandas as pd
import torch
from ows_retrieval.data import load_docs, load_doc_embeddings
from ows_retrieval.jina_v3_encoder import load_jina_v3

OUT_PATH = os.path.join(os.path.dirname(__file__), 'probe_pipeline_mismatch.out')
out_file = open(OUT_PATH, 'w', encoding='utf-8')


def log(msg=''):
    print(msg)
    out_file.write(msg + '\n')
    out_file.flush()


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f"using device: {DEVICE}")
log()

docs = load_docs()
emb = load_doc_embeddings()
docs_by_id = docs.set_index('id')

# Collect 10 single-chunk docs with non-zero stored vectors.
samples = []
for _, row in emb.iterrows():
    if len(row['embeddings']) != 1:
        continue
    v = np.asarray(row['embeddings'][0], dtype=np.float32)
    if np.linalg.norm(v) == 0:
        continue
    samples.append(row)
    if len(samples) == 10:
        break

# Pre-extract chunk text and unit-normalized stored vector for each sample.
texts = []
stored_unit = []
for row in samples:
    offsets = row['chunk_offsets'][0]
    start, end = int(offsets['start']), int(offsets['end'])
    main = docs_by_id.loc[row['id'], 'main_content']
    main = '' if pd.isna(main) else str(main)
    texts.append(main[start:end])
    v = np.asarray(row['embeddings'][0], dtype=np.float32)
    stored_unit.append(v / np.linalg.norm(v))
stored_unit = np.stack(stored_unit)

model = load_jina_v3().to(DEVICE)

TASKS = [
    'retrieval.query',
    'retrieval.passage',
    'text-matching',
    'separation',
    None,
]


def encode(text, task):
    with torch.no_grad():
        v = model.encode([text], task=task) if task else model.encode([text])
    v = np.asarray(v, dtype=np.float32)
    return (v / np.linalg.norm(v, axis=1, keepdims=True))[0]


# Test A: per-doc, per-task self-match cosine.
log("Test A: cosine(our re-encode of doc i, stored vector of doc i)")
log("  one row per doc, one column per task; 'best' = max across tasks")
log()
header = f"{'doc id':18s}  " + "  ".join(f"{(t or 'no-task'):>17s}" for t in TASKS) + "    best"
log(header)
log("-" * len(header))

best_cosines = []
# For test B we cache per-doc encodings under no-task (the best column from
# the previous run), so we can build the cross-doc matrix without recomputing.
no_task_vecs = []

for i, (row, text) in enumerate(zip(samples, texts)):
    if not text.strip():
        continue
    row_cosines = []
    for task in TASKS:
        v = encode(text, task)
        c = float(v @ stored_unit[i])
        row_cosines.append(c)
        if task is None:
            no_task_vecs.append(v)
    best = max(row_cosines)
    best_cosines.append(best)
    row_str = "  ".join(f"{c:17.4f}" for c in row_cosines)
    log(f"{row['id'][:16]}..  {row_str}    {best:.4f}")

log()
log(f"mean best self-match (any task): {np.mean(best_cosines):.4f}")
log(f"max  best self-match (any task): {np.max(best_cosines):.4f}")
log()

# Test B: cross-doc baseline in no-task mode.
# Build the N x N cosine matrix: rows = our no-task encodings, cols = stored.
no_task_vecs = np.stack(no_task_vecs)
xcos = no_task_vecs @ stored_unit.T  # shape (N, N)

log("Test B: cross-doc matrix in NO-TASK mode")
log("  rows = our no-task encoding of doc i")
log("  cols = stored vector of doc j")
log("  diagonal = self-match;  off-diagonal = match against other docs")
log()
log("       " + "  ".join(f"stored_{j}" for j in range(len(samples))))
for i, vrow in enumerate(xcos):
    log(f"ours_{i} " + "  ".join(f"{c:8.4f}" for c in vrow))

diag = np.diag(xcos)
mask = ~np.eye(len(samples), dtype=bool)
off = xcos[mask]

log()
log(f"diagonal     (self-match)  mean = {diag.mean():.4f}   min = {diag.min():.4f}   max = {diag.max():.4f}")
log(f"off-diagonal (random pair) mean = {off.mean():.4f}   min = {off.min():.4f}   max = {off.max():.4f}")
log(f"gap (diag - off mean)             = {diag.mean() - off.mean():+.4f}")
log()
log("Interpretation:")
log("  - If diag - off >> 0:  no-task IS discriminative, and ~0.88 self-match")
log("    means the upstream really did use no-task (or very close). In that")
log("    case, encoding queries with no-task and using the stored vectors")
log("    could work without re-embedding.")
log("  - If diag - off ~ 0:   no-task is non-discriminative; 0.88 is just")
log("    the baseline cosine of any two no-task vectors. Re-embedding (with")
log("    a proper query/passage task pair) remains the right answer.")

out_file.close()
print(f"\nlog written to {OUT_PATH}")
