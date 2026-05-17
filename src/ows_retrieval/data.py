"""Loaders for the OWS Curlie / Radboud-val data files."""
from pathlib import Path

import pandas as pd

# Path to the data/raw folder, relative to this file.
# __file__ is the path of this script (data.py).
# Going up three parents: ows_retrieval/ -> src/ -> project root.
DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


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
