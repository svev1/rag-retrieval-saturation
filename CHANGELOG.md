# Changelog

Notable milestones for this project. This is a high-level record, not a
replacement for the git history.

### Project history

- **M1** — dataset loaded; BM25 and dense (Jina v5-nano + FAISS) retrieval;
  Recall@k / nDCG@k baselines.
- **M2** — RAG generation pipeline and cached k-sweep answers (Gemma 4 via Ollama).
- **M3** — RAGAS Faithfulness / Answer Relevancy scoring, blind manual audit, and
  the retrieval-noise vs. generation-quality analysis.
- Re-embedded documents and queries with a pinned Jina v5-nano checkpoint instead
  of the provided Jina V3 embeddings, to keep document/query encoding consistent
  and reproducible.
