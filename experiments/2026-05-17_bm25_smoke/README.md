# 2026-05-17 BM25 smoke

A sanity-check run of the BM25 pipeline on a small slice of the corpus,
before scaling to the full 63k documents. No metrics are computed here.
The point of this run was to confirm by hand-eyeball that top-10
results look topical for each query.

## What was run

- Smoke slice: 4 query topics chosen for diversity of difficulty.
  - 24 "Iceland nature": broad informational, 41 relevant docs.
  - 31 "decision trees": short factual, 22 relevant docs.
  - 32 "Hyperion Dan Simmons review": named entity, 16 relevant docs.
  - 52 "Protein shake before or after workout": sparse positives (only
    5 relevant docs out of 50 judged), wordy with stopwords.
- Corpus: all qrels-judged docs for the 4 queries plus 500 random
  unjudged documents as distractors. Total around 691 docs.
- BM25: `rank_bm25.BM25Okapi` with default parameters, NLTK English
  stopword removal, lowercase word tokenization on the regex `\w+`.
- Indexed text per doc: `title + description + main_content`.

## What I looked for

The plan said the smoke is verified if each query gets at least one
judged-relevant doc in the top-10 and the titles look topical.

## What I observed

All four queries returned topical top-10 results:

- Iceland nature: 8 of 10 results judged relevant, mostly Iceland
  travel and government nature pages.
- decision trees: 6 of 10 judged relevant, mostly ML and decision
  tree articles. Two false positives matched on "tree" or "decision"
  in unrelated contexts, which is expected behaviour for keyword BM25.
- Hyperion Dan Simmons review: most top results are Dan Simmons
  novels and author pages. Some near-duplicate documents appear in
  the corpus with the same content but different relevance labels.
- Protein shake before or after workout: 2 of 10 judged relevant, top
  results are protein and workout content. This query has only 5
  relevant docs in qrels, so 2 in top-10 is acceptable.

Conclusion: BM25 pipeline behaves as expected on the smoke slice. We
can scale to the full corpus.

## Files

- `config.yaml`: smoke parameters (query ids, distractor count, seed).
- `results.json`: top-10 for each query, with rank, score, doc_id,
  title, and qrels relevance label (null if unjudged).
- `README.md`: this file.

## Reproducibility

To reproduce `results.json`:

```bash
python -c "from ows_retrieval.report import smoke_experiment; smoke_experiment([24, 31, 32, 52])"
```
