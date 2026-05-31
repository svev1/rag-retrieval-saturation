"""Loaders for the OWS Curlie / Radboud-val data files."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

# Jina V5 nano re-embedded artifacts (see docs/embeddings.md). One row per
# chunk for docs, one row per query. Embeddings are L2-normalized float32.
CHUNK_EMB_FILE = "radboud-val.jina-v5-nano.chunks-1024.parquet"
QUERY_EMB_FILE = "radboud-val.jina-v5-nano.queries.parquet"

def load_topics() -> pd.DataFrame:
    """Load the topics from topics.tsv.

    The file is tab-separated, with no header. Columns are the topic id
    and the query text.
    """
    return pd.read_csv(
        DATA_RAW / "topics.tsv",
        sep="\t",
        header=None,
        names=["topic_id", "query"],
    )


def load_qrels() -> pd.DataFrame:
    """Load the relevance judgments from qrels.txt.

    TREC qrels format: space-separated, no header, four columns:
    topic_id, "Q0" (a TREC convention we don't need), doc_id, rel.
    Returns three columns: topic_id, doc_id, rel.
    """
    qrels = pd.read_csv(
        DATA_RAW / "qrels.txt",
        sep=" ",
        header=None,
        names=["topic_id", "_q0", "doc_id", "rel"],
    )
    return qrels.drop(columns=["_q0"])


# Columns we actually need from the docs parquet.
# The full file has 48 columns; loading only what we use keeps memory low.
# "id" is the SHA-256 hex string that matches doc_id in qrels.
DOC_COLUMNS = ["id", "title", "description", "main_content", "url"]


def load_docs() -> pd.DataFrame:
    """Load the documents from subsample.parquet.

    Returns a DataFrame with columns: id, title, description, main_content, url.
    """
    return pd.read_parquet(DATA_RAW / "subsample.parquet", columns=DOC_COLUMNS)


def load_doc_embeddings() -> pd.DataFrame:
    """Load the precomputed Jina V3 embeddings from subsample_embeddings.parquet.

    Each document was split into chunks before embedding, so the 'embeddings'
    column contains a list of vectors per document (one per chunk), not a
    single vector. Aggregation (mean / max / etc.) is decided later when
    building the FAISS index.

    Returns columns: id, chunk_offsets, embeddings.
    """
    return pd.read_parquet(
        DATA_RAW / "subsample_embeddings.parquet",
        columns=["id", "chunk_offsets", "embeddings"],
    )


def _load_embedding_matrix(path: Path, id_column: str) -> tuple[np.ndarray, np.ndarray]:
    """Read an (id, embedding) parquet into a dense matrix and a parallel id array.

    The 'embedding' column is a list<float>[dim] per row. We flatten the whole
    column once and reshape to (n_rows, dim) — much faster than building the
    matrix row by row.
    """
    table = pq.read_table(path, columns=[id_column, "embedding"])
    ids = table.column(id_column).to_numpy(zero_copy_only=False)
    embedding = table.column("embedding").combine_chunks()
    dim = len(embedding[0])
    vectors = embedding.flatten().to_numpy(zero_copy_only=False)
    vectors = vectors.reshape(len(ids), dim).astype(np.float32)
    return vectors, ids


def load_chunk_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Load the V5 document-chunk embeddings.

    Returns:
        vectors: (n_chunks, 768) float32 matrix, one row per chunk.
        doc_ids: (n_chunks,) array mapping each chunk row to its document id.

    Doc-level scoring aggregates chunk scores per doc_id later (max-over-chunks).
    """
    return _load_embedding_matrix(DATA_RAW / CHUNK_EMB_FILE, "doc_id")


def load_query_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Load the V5 query embeddings.

    Returns:
        vectors: (n_queries, 768) float32 matrix, one row per query.
        query_ids: (n_queries,) array of query ids (these equal topic_id).
    """
    return _load_embedding_matrix(DATA_RAW / QUERY_EMB_FILE, "query_id")
