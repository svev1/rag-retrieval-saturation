# 2026-05-29 V5 embedding sanity

Does dense retrieval **work** with the re-embedded Jina V5 nano vectors? This is
the "after" bookend to `2026-05-25_dense-embedding-investigation` (which showed
the shipped V3 vectors were broken for retrieval).

## What it checks

For 5 topics that have relevant docs in qrels, `v5_sanity.py`:

1. looks up the topic's **query** vector in `radboud-val.jina-v5-nano.queries.parquet`,
2. for up to 3 **relevant** docs (per qrels), fetches all their chunks from
   `radboud-val.jina-v5-nano.chunks-1024.parquet` and takes the **max cosine over
   chunks** against the query,
3. does the same for 3 **random** unjudged docs,
4. reports relevant-vs-random cosines per topic and overall.

Both sides are pre-normalized, so cosine = inner product. No model is loaded —
this reads the precomputed parquets only.

**Pass criterion:** the relevant–random gap is clearly positive and every
relevant cosine beats the random max.

## Result (see `v5_sanity.out`)

```
relevant : n=15  mean=0.6665  min=0.3752  max=0.8101
random   : n=15  mean=0.0871  min=-0.0048  max=0.1485
gap (rel mean - rand mean) = +0.5794
```

Relevant docs sit at ~0.67 cosine, random docs at ~0.09 — a clean +0.58
separation, with every relevant doc above every random one. Contrast the
investigation, where the V3 vectors buried relevant docs at rank ~1,900. The V5
re-embedding gives the query↔passage separation dense retrieval needs.

## Reproduce

```powershell
.venv\Scripts\python.exe experiments\2026-05-29_v5-embedding-sanity\v5_sanity.py
```

Run from the repo root (the data paths are relative). Writes `v5_sanity.out`
next to the script.
