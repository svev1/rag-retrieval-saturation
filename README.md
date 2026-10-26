# Do More Retrieved Web Documents Improve RAG? Retrieval Saturation on the Open Web Search Curlie 2025 Collection

Seminar project (Data Science Seminar, University of Passau) studying whether
**retrieval saturation** — the point where adding more retrieved documents stops
improving a RAG system's answers — occurs on noisy, web-derived data, and whether
it differs between sparse and dense retrieval.

**Corpus:** the `ows-curlie-2025/radboud-val` split of the Open Web Search Curlie
2025 collection. **Retrievers:** BM25 (sparse) and Jina v5-nano embeddings +
FAISS (dense). **Generator:** Gemma 4 (`gemma4:e4b`) via Ollama. **Evaluation:**
RAGAS Faithfulness and Answer Relevancy, calibrated against a blind manual audit,
plus a qrels-based retrieval-noise analysis.

## Research questions

- **RQ1** — Does retrieval saturation occur on `radboud-val`?
- **RQ2** — Does saturation differ between sparse and dense retrieval?
- **RQ3** — Is lower generation quality associated with noisier retrieved contexts?

## Main findings

- Generation quality **saturates early** (answer relevancy is best at ~3 documents,
  then plateaus); faithfulness is high and flat.
- BM25 outperforms dense retrieval at every `k` on Recall@k and nDCG@k.
- Retrieval noise rises with `k`, but the noise–quality Spearman correlation is
  ≈ 0 — more (mostly unjudged) context neither helps nor hurts on this corpus.

See the [final seminar paper (PDF)](docs/reports/nemeth-alice-sveva-seminar-paper.pdf)
for the full write-up.

## Repository layout

- `src/ows_retrieval/` — the pipeline: embedding, BM25 and dense retrieval,
  RAG generation (`rag/`), and RAGAS evaluation (`rageval/`).
- `configs/` — the embedding job configuration.
- `experiments/` — one folder per run (config, README, results, plots). The
  reported evaluation is `experiments/2026-06-13_rag_eval_clean-index/`.
- `docs/` — methodology, embeddings notes, and `reports/` for the deliverables.

## Reproducing the results

Dependencies are managed with `pyproject.toml`:

```bash
uv sync           # or: pip install -e .
```

Document access uses the `ir-datasets-owi` package and requires the OWI
`ows-curlie-2025` collection available locally (see `src/ows_retrieval/embedding/cli.py`
for the dataset-link command). Each `experiments/*/README.md` documents the exact
command that produced that run; the high-level pipeline is:

1. **Embeddings** — `configs/radboud-val.toml` drives the embedding job
   (see [`docs/embeddings.md`](docs/embeddings.md)).
2. **Retrieval** — BM25 and dense k-sweeps produce TREC run files and
   Recall@k / nDCG@k (via `ir-measures`).
3. **Generation** — `python -m ows_retrieval.rag.sweep <experiment>/config.yaml`
   caches one answer per (retriever, topic, k).
   The answer caches (`experiments/*/cache/`) are not included in this
   repository: they embed the full text of the retrieved web pages, which
   belongs to the respective page owners and may not be redistributed under
   the Open Web Index Licence. Re-run this step to regenerate them.
4. **Evaluation** — `ows_retrieval.rageval` collates, scores with RAGAS, computes
   noise, and synthesises the plots and summary.

## Use of AI Tools

A generative AI was used in this project. Its
use is disclosed for transparency:

- **Code (`src/`):** assistance with writing and refactoring pipeline code,
  docstrings and documentation, and debugging. 
- **Report:** help with LaTeX and ACM formatting, polishing the writing, and generating and relabeling the figures.

The research design, experiments, evaluation, and interpretation of the results
are the author's own work.

## License

MIT — see [`LICENSE`](LICENSE). The MIT license covers the code in this
repository only, not the dataset or the web content it indexes.

## Data acknowledgement

This work uses index files as part of the index partition created by the
OpenWebSearch.eu project that has received funding from the European Union's
Horizon Europe research and innovation programme under grant agreement
No 101070014 (OpenWebSearch.EU). The data is used under the
[Open Web Index Licence (OWIL)](https://openwebsearch.eu/owil-current/).
