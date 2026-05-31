"""Even when our query side matches the upstream's task choice for docs,
how well do relevant documents (per qrels) rank in the dense pipeline?

Builds the full FAISS index over the stored doc vectors, encodes each
sample topic's query with each Jina V3 task mode (including no-task),
and reports the rank of every qrels-relevant doc against the resulting
ranking. If retrieval is broken even when the query-side task matches
the doc-side task, no choice of query task can rescue the pipeline.

Pairs with probe_pipeline_mismatch.py: that probe showed the upstream
used no-task; this one shows no-task on the query side still doesn't
rescue ranking, so the upstream pipeline choice itself is the problem
and re-embedding with proper asymmetric tasks is the right fix.

Writes a UTF-8 log next to itself (probe_alignment.out)."""
import os
import numpy as np
from ows_retrieval.data import load_topics, load_qrels, load_doc_embeddings
from ows_retrieval.dense import flatten_chunks, build_index
from ows_retrieval.jina_v3_encoder import load_jina_v3

OUT_PATH = os.path.join(os.path.dirname(__file__), 'probe_alignment.out')
out_file = open(OUT_PATH, 'w', encoding='utf-8')


def log(msg=''):
    print(msg)
    out_file.write(msg + '\n')
    out_file.flush()


log("Building index...")
emb = load_doc_embeddings()
chunk_vectors, chunk_to_doc = flatten_chunks(emb)
index = build_index(chunk_vectors)
log(f"  index.ntotal = {index.ntotal}")
log()

topics = load_topics()
qrels = load_qrels()

sample_tids = qrels[qrels['rel'] > 0]['topic_id'].drop_duplicates().head(3).tolist()
model = load_jina_v3()


def encode_with_task(text, task):
    if task is None:
        v = model.encode([text])
    else:
        v = model.encode([text], task=task)
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


TASKS = [
    'retrieval.query',
    'retrieval.passage',
    'text-matching',
    'separation',
    None,
]
N_CHUNKS = 20000

for tid in sample_tids:
    query = topics[topics['topic_id'] == tid].iloc[0]['query']
    rel = qrels[(qrels['topic_id'] == tid) & (qrels['rel'] > 0)]
    rel_doc_ids = set(rel['doc_id'].tolist())

    log(f"Topic {tid}: {query!r}   ({len(rel_doc_ids)} relevant docs)")
    log(f"  {'task':22s}  {'best_rel_rank':>13s}  {'median_rel_rank':>15s}  {'rels_in_topN':>13s}")

    for task in TASKS:
        q_vec = encode_with_task(query, task)
        scores, idxs = index.search(q_vec, N_CHUNKS)

        doc_best = {}
        for s, i in zip(scores[0], idxs[0]):
            d = chunk_to_doc[i]
            if d not in doc_best or s > doc_best[d]:
                doc_best[d] = float(s)

        ranked = sorted(doc_best.items(), key=lambda x: -x[1])
        rank_of = {d: r + 1 for r, (d, _) in enumerate(ranked)}

        rel_ranks = [rank_of.get(d, N_CHUNKS + 1) for d in rel_doc_ids]
        in_topN = sum(r <= N_CHUNKS for r in rel_ranks)
        best = min(rel_ranks)
        median = int(np.median(rel_ranks))
        task_str = task if task else '(no task)'
        log(f"  {task_str:22s}  {best:>13d}  {median:>15d}  {in_topN:>3d}/{len(rel_doc_ids)}")
    log()

out_file.close()
print(f"\nlog written to {OUT_PATH}")
