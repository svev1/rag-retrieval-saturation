# 2026-05-25 Dense embedding investigation

Why we are **not** using `data/raw/subsample_embeddings.parquet` (the
precomputed Jina V3 doc vectors shipped with the dataset), and instead
re-embed the corpus ourselves with Jina V5 nano.

## TL;DR

The shipped doc vectors don't give working dense retrieval. The
investigation showed two things:

1. **Pipelines mostly match.** The upstream embedded docs with Jina V3
   in *no-task* mode (`model.encode(text)` with no task argument). We
   can reproduce that with self-match cosine ~0.88.
2. **But no-task is the wrong mode for retrieval.** Even when we
   encode queries in no-task too, relevant docs land at rank
   ~1 900–6 500 out of 60 k. No-task lacks the query↔passage
   asymmetric training that's needed for short-query → long-passage
   matching.

→ Re-embed with the proper asymmetric task pair
(`prompt_name="document"` for chunks, `prompt_name="query"` for
queries).

## The probes

Two scripts in this folder, each writing its own UTF-8 `.out` log.

### `probe_alignment.py` — does retrieval work?

Builds the FAISS index over the stored V3 vectors and, for 3 sample
topics, asks where the qrels-relevant docs rank when the query is
encoded with each Jina V3 task mode.

| Topic | Best query task | Best rel. rank | Median rel. rank |
|---|---|---:|---:|
| 3 (split ergo keyboard) | `separation` | 1 730 | > 20 000 |
| 4 (metoo Hollywood) | `retrieval.passage` | 1 395 | > 20 000 |
| 7 (gastritis symptoms) | `text-matching` | 3 476 | 11 998 |

No task choice makes retrieval work, and the "best" choice differs
per topic.

### `probe_pipeline_mismatch.py` — which task did the upstream use?

For 10 single-chunk docs, re-encodes the chunk text with each task
mode and computes cosine against the stored vector. Then runs a
10×10 cross-doc matrix in the best mode to check the result isn't
just a baseline artefact.

| Task | Mean self-match cosine vs stored |
|---|---:|
| `retrieval.query` | 0.58 |
| `retrieval.passage` | 0.49 |
| `text-matching` | 0.65 |
| `separation` | 0.84 |
| **no task** | **0.88** |

In the no-task 10×10 matrix: diagonal mean 0.88, off-diagonal mean
0.45, gap **+0.43**. Every diagonal entry is the maximum in its row,
so the match is real — the upstream did use no-task.

### Why the two probes need each other

`probe_pipeline_mismatch` reveals the upstream's mode. On its own
that would suggest "use no-task on the query side and you're fine."
`probe_alignment` rules that out: it shows that even matching the
upstream's no-task choice on the query side still puts relevant docs
at rank 1 900+. So the problem is the upstream's *choice itself*, not
a mismatch we can paper over.

## What we did instead

We decided the best option was to embed the documents ourselves,
re-embedding the `radboud-val` split with a smaller, retrieval-aligned
model:

- Model: `jinaai/jina-embeddings-v5-text-nano-retrieval`,
  revision `ac5d898c8d382b17167c33e5c8af644a3519b47d`.
- Dim: 768. Both sides L2-normalized. Asymmetric task pair via
  `prompt_name`: `document` for chunks, `query` for queries.
- 1024-token chunks with 128-token overlap. 302 736 chunks across
  63 639 docs. Both chunk and query embeddings are pre-computed.

Artifacts in `data/raw/`:

- `radboud-val.jina-v5-nano.chunks-1024.parquet`
- `radboud-val.jina-v5-nano.queries.parquet`

Full embedding pipeline config: `docs/embeddings.md`.

The precomputed V3 file (`subsample_embeddings.parquet`) is left in
place but unused.

## Reproducibility

The captured `.out` files next to each probe are the durable record.
To re-run the probes, you need the Jina V3 model at runtime. The
loader lives in `src/ows_retrieval/jina_v3_encoder.py`, and the heavy
deps it needs (`transformers`, `torch`, `einops`, `sentencepiece`) are
in the `investigation` optional-dependency group rather than the core
deps, since the dense baseline itself doesn't need them. Install them
once, then run from the repo root:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[investigation]"
.venv\Scripts\python.exe experiments\2026-05-25_dense-embedding-investigation\probe_pipeline_mismatch.py
.venv\Scripts\python.exe experiments\2026-05-25_dense-embedding-investigation\probe_alignment.py
```
