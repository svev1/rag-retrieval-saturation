# RAG answer evaluation (clean-index sweep, gemma4 generator)

Evaluation of the cached RAG answers. Code: `src/ows_retrieval/rageval/`.

## dataset.parquet — the answers to score

One row per (retriever, topic, k) answer; `contexts` is the per-document text the
generator saw (whitespace collapsed, truncated to char_cap=500). Built from the
gemma4 sweep cache:

```
python -m ows_retrieval.rageval.collate \
  experiments/2026-06-13_rag_ksweep_clean-index_gemma4/cache \
  experiments/2026-06-13_rag_eval_clean-index/dataset.parquet \
  --doc-text-source clean --char-cap 500
```

## noise.parquet — retrieval noise from qrels (no LLM)

One row per (retriever, topic, k): relevant/non-relevant/unjudged counts and
`noise` (incl. unjudged) / `noise_strict` (confirmed non-relevant only).

```
python -m ows_retrieval.rageval.noise experiments/2026-06-13_rag_eval_clean-index/noise.parquet
```

## scores.parquet — RAGAS faithfulness + answer relevancy

Local judge (gemma4:e4b) + embedder (nomic-embed-text). Resumable: appends to a
`scores.jsonl` log and skips keys already done.
```
# quick check on a few rows first:
python -m ows_retrieval.rageval.score \
  experiments/2026-06-13_rag_eval_clean-index/dataset.parquet \
  experiments/2026-06-13_rag_eval_clean-index/scores.parquet --limit 3

# full run (resumes; skips the rows already done):
python -m ows_retrieval.rageval.score \
  experiments/2026-06-13_rag_eval_clean-index/dataset.parquet \
  experiments/2026-06-13_rag_eval_clean-index/scores.parquet

# re-print the summary later without re-scoring:
python -m ows_retrieval.rageval.score x scores.parquet --summary-only
```

faithfulness is NaN for refusals (expected, reported separately); a NaN on a
non-refusal is a judge glitch — the summary lists those rows; delete their lines
from `scores.jsonl` and re-run to re-score them. (1 documented artifact: dense t43
k20 emits malformed JSON; all 10 claims judged supported → effectively F=1.0.)

## synthesis — join scores x noise (RQ3)

```
python -m ows_retrieval.rageval.synthesis \
  experiments/2026-06-13_rag_eval_clean-index/scores.parquet \
  experiments/2026-06-13_rag_eval_clean-index/noise.parquet
```

Writes `summary.csv` (mean quality + noise per retriever x k) and
`quality_vs_k.png` / `noise_vs_k.png`. Spearman(noise, quality) ~= 0: quality is
flat vs noise; answer relevancy saturates by k~3; faithfulness high & flat.

## manual audit (blind)

`audit.py` draws a deterministic stratified sample of 24 answers (2 per retriever
x k, seed 42) and writes `audit_sheet.xlsx`: question, answer, documents,
and empty Yes/Partly/No columns. The retriever/topic/k keys are hidden so the rating
is not anchored, and **no RAGAS scores are present** — the audit is done blind.

```
python -m ows_retrieval.rageval.audit \
  experiments/2026-06-13_rag_eval_clean-index/dataset.parquet \
  experiments/2026-06-13_rag_eval_clean-index/audit_sheet.xlsx
```

The committed `audit_sheet.xlsx` is the **filled** version: `manual_faithfulness`,
`manual_relevancy`, and `notes` rated by hand from the question, answer, and
documents — still no RAGAS.

## agreement (manual vs RAGAS)

Only here do the blind ratings meet the RAGAS scores. `agreement.py` joins the filled
sheet with `scores.parquet` on (retriever, topic_id, k):

```
python -m ows_retrieval.rageval.agreement \
  experiments/2026-06-13_rag_eval_clean-index/audit_sheet.xlsx \
  experiments/2026-06-13_rag_eval_clean-index/scores.parquet
```

Writes `audit_agreement.csv`. Result: faithfulness — human Yes on all 24, RAGAS
mean 0.98 (min 0.80) → strong agreement. Relevancy — RAGAS mean 0.82 (human "Yes")
vs 0.69 (human "Partly"), Spearman +0.48 → moderate agreement, RAGAS slightly
lenient on borderline cases.
