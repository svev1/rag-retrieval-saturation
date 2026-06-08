# 2026-06-07 RAG k-sweep — full cache (html-index, v1)

The deliverable answer cache for M3. For every retriever × topic × k we take the
top-k retrieved docs, build the fixed-template prompt, generate with local Gemma,
and cache the answer with provenance. M3 scores these with RAGAS without
regenerating.

This is **v1 ("html-index")**, the investigation run: the sparse (bm25) run files
were built over HTML-laden text (`title+description+main_content`). A later
**v2 ("clean-index")** will rebuild bm25 over `default_text()` and rerun unchanged,
so the two caches isolate the effect of that indexing change.

## What was run

- All **28 topics** × 2 retrievers (bm25, dense) × 6 k `{1,3,5,10,20,50}` =
  **336 cached answers**.
- Generator: `gemma2:9b` via Ollama, deterministic (`temperature=0`, `seed=42`,
  `num_ctx=8192`, `num_predict=512`). `num_gpu=24` (partial GPU offload: full
  offload OOMs `9b` + the 8k KV cache on this 8 GB card).
- Context: per-doc front-truncation to 500 chars, HTML stripped for the prompt only.
- See `config.yaml` for the exact parameters.

## What I looked for

- All 336 keys generate (resumable, crash-tolerant).
- No silent context truncation (every prompt ≤ `num_ctx − num_predict = 7680`).
- No answer cutoffs (output tokens < `num_predict = 512`).
- Eyeball for anomalies across topics and k.

## What I observed

- **336/336 generated, 0 failed, 0 skipped.** 28 per-topic folders.
- Token budget held: max prompt **7558** tok < 7680 (every k=50 fit). Max output
  **185** tok, no cutoffs. No empty answers.
- **86/336 (26%) are the instructed refusal** *"The documents do not answer this."*
  (`output_tokens=10`). Two patterns, both expected behaviour, not bugs:
  1. **Context dilution at k=50.** Some topics answer at low k but refuse at k=50
     even though the answer-bearing doc is still present (nested prefix) — it gets
     buried among 50 front-truncated, noisy snippets. e.g. topic 23 "excel sum
     cells" answers at k=5 (`=SUM(A2:A20)`) but refuses at k=50. k=50 is the
     single largest refusal bucket. This is the "more context can hurt" RAG effect.
  2. **Hard/niche topics.** Topics 15, 20, 44, 52, 54 refuse across most/all k for
     one or both retrievers — specific informational queries a general web crawl
     plausibly does not directly answer.

Note for M3: a refusal is the model following the frozen v1 instruction, not an
unfaithful answer. Where the supporting doc IS present (topic 23 / k=50), the
refusal is a low answer-relevance signal — a quality dimension RAGAS should
interpret, and a candidate motivation for prompt tuning in a later pass.

## Files

- `config.yaml`: full-sweep parameters.
- `cache/t{topic}/{retriever}__k{k}.json`: 336 answer records, one per key, each
  with answer, prompt, doc_ids, token counts, and provenance.
- `README.md`: this file.

## Reproducibility

```bash
python -m ows_retrieval.rag.sweep experiments/2026-06-07_rag_ksweep_html-index/config.yaml
```
