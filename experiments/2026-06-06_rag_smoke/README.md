# 2026-06-06 RAG k-sweep smoke

A sanity-check of the M2 RAG pipeline (retrieve → context → prompt → Gemma →
cache) before the full sweep. Confirms it runs end-to-end and the answers are
non-empty and topical. No scoring here (that is M3).

## What was run

- 4 smoke topics `{24, 31, 32, 52}` × 2 retrievers (bm25, dense) × 6 k
  `{1,3,5,10,20,50}` = **48 cached answers**.
- Generator: `gemma2:2b` via Ollama, deterministic (`temperature=0`, `seed=42`,
  `num_ctx=8192`, `num_predict=512`). Per-doc context front-truncated to 500 chars,
  HTML stripped for the prompt only.
- See `config.yaml` for the full parameters.

## What I looked for

- All 48 keys generate (resumable, skip-if-exists).
- Token budget holds: every prompt fits `num_ctx − num_predict = 7680` tokens, so
  no silent context truncation; output tokens stay under the 512 cap (no cutoffs).
- Hand-eyeball: answers non-empty, topical, cite `[Document N]`, richer at higher k.

## What I observed

- 48/48 generated, 0 skipped. Largest prompt 6634 tok (dense, t32, k50) < 7680 —
  fits with margin. Output tokens 10–263, none at the 512 cap.
- Answers are grounded and cite documents inline. The expected pattern shows:
  thin at k=1, richer synthesis at k=10.
- At k=50 Gemma occasionally adds outside knowledge (specific names not in the
  truncated snippets) despite the "use ONLY the documents" instruction — a
  faithfulness signal for M3 to quantify.

Conclusion: pipeline behaves as expected. Ready to pick the evaluation subset and
run the full sweep.

## Files

- `config.yaml`: smoke parameters.
- `cache/`: 48 answer records, one JSON per `(retriever, topic, k)`, each with the
  answer, prompt, doc_ids, token counts, and provenance.
- `README.md`: this file.

## Reproducibility

```bash
python -m ows_retrieval.rag.sweep experiments/2026-06-06_rag_smoke/config.yaml
```
