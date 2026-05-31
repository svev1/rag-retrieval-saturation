# Re-embedded radboud-val artifacts

We re-embedded the `radboud-val` split ourselves with a smaller,
retrieval-aligned model. The final parquet artifacts live in `data/raw/`:

- `data/raw/radboud-val.jina-v5-nano.chunks-1024.parquet` — document chunk embeddings
- `data/raw/radboud-val.jina-v5-nano.queries.parquet` — query embeddings

## Embedding parameters (identical for both files)

- **Model:** `jinaai/jina-embeddings-v5-text-nano-retrieval`
- **Model revision (HF commit pin):** `ac5d898c8d382b17167c33e5c8af644a3519b47d`
- **`trust_remote_code`:** true
- **Embedding dim:** 768
- **Output dtype:** float32
- **Normalized:** true (L2; observed norms ≈ 1.0 ± ~3e-3 from float32 rounding — re-normalize before cosine if you want exactness)
- **`embedding_task`:** `""` (empty — task selection is driven by `prompt_name`, not by Jina's task arg)
- **Prompt name:** `document` for doc chunks, `query` for queries
- **Source:** `ows-curlie-2025/radboud-val`, text field `default_text` (i.e. `doc.default_text()`)

## Document chunking

- 1024-token windows with 128-token overlap
- Tokenizer is the Jina v5 model tokenizer
- 302,736 chunks across 63,639 unique docs (median chunk = 1024 tokens; tail chunks can be as small as 1)
- The artifact does **not** store chunk text. To recover text: load the doc by `doc_id`, normalize line endings + surrounding whitespace, slice `char_start:char_end`, verify against `chunk_text_sha256`.

## Parquet schemas

**chunks (302,736 rows):** `chunk_id, doc_id, chunk_index, token_start, token_end, token_count, char_start, char_end, chunk_text_sha256, source_dataset, source_subset, source_text_field, model_id, model_revision, embedding_task, prompt_name, normalized, output_dtype, embedding_dim, embedding (list<float>[768])`

**queries (28 rows):** `query_id, query_index, query_text_sha256, source_dataset, source_subset, source_text_field, model_id, model_revision, embedding_task, prompt_name, normalized, output_dtype, embedding_dim, embedding (list<float>[768])`

## Retrieval notes

- Embeddings are L2-normalized → use **inner product** (cosine = dot) for ranking.
- Doc-level scoring requires aggregating chunk scores per `doc_id` (e.g. max-over-chunks).
- Pinned model revision + `chunk_text_sha256` / `query_text_sha256` give full provenance — keep both pins (revision + chunking config) if you re-embed anything new.

## Source config snapshot

- `chunk_size_tokens = 1024`, `chunk_overlap_tokens = 128`
- `encode_batch_size = 16`, `write_batch_size = 512`
- `device = "auto"`

## Reproducing these artifacts

The embedding pipeline lives in `src/ows_retrieval/embedding/` and is driven by
`configs/radboud-val.toml`. The retrieval baselines do **not** need any of this —
only re-run it to regenerate the parquet files.

```powershell
# 1. Install the heavy embedding deps (separate from the core baseline).
.venv\Scripts\python.exe -m pip install -e ".[embedding]"

# 2. (Optional) Re-pin the model revision before a full run.
.venv\Scripts\python.exe -m ows_retrieval.embedding.resolve_revision jinaai/jina-embeddings-v5-text-nano-retrieval

# 3. Smoke run (first 100 docs / queries) → writes *.smoke.parquet, won't clobber the real files.
ows-embed --config configs/radboud-val.toml --limit-docs 100
ows-embed-queries --config configs/radboud-val.toml --limit-docs 100

# 4. Full run → writes the canonical artifacts into data/raw/.
ows-embed --config configs/radboud-val.toml
ows-embed-queries --config configs/radboud-val.toml

# 5. Verify an artifact.
ows-verify-embeddings data/raw/radboud-val.jina-v5-nano.chunks-1024.parquet --kind docs
```

**Requirements / caveats:**

- The job reads documents through `ir_datasets_owi`, which needs the OWI
  `ows-curlie-2025` collection available locally and linked for `ir-datasets`.
  Without that data the job cannot load the corpus.
- Embeddings are not guaranteed bit-identical across hardware (different CPU /
  GPU backends produce tiny float differences). The `chunk_text_sha256` /
  `query_text_sha256` hashes verify that the *text* being embedded matches; the
  vectors are equivalent, not byte-equal.
