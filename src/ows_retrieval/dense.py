"""Dense retrieval index over precomputed Jina V3 chunk embeddings.

The documents were embedded chunk-by-chunk, so each document contributes
multiple vectors. We flatten all chunks into one matrix, L2-normalize each
row, and store them in a FAISS IndexFlatIP. Inner product on unit vectors
equals cosine similarity, so the index returns chunks ranked by cosine.

Mapping from FAISS row index back to document id is held in a parallel
array (chunk_to_doc), mirroring the docs.iloc[pos] trick used in bm25.py.
"""
import faiss
import numpy as np
import pandas as pd


def flatten_chunks(emb_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Unroll the per-document chunk lists into a flat, normalized matrix."""
    vectors = []
    doc_ids = []
    for doc_id, chunks in zip(emb_df["id"], emb_df["embeddings"]):
        for chunk in chunks:
            vectors.append(chunk)
            doc_ids.append(doc_id)

    # Stack into one matrix and cast: FAISS only accepts float32.
    chunk_vectors = np.asarray(vectors, dtype=np.float32)
    chunk_to_doc = np.asarray(doc_ids, dtype=object)

    # Drop zero-norm rows. A small fraction of chunks are all-zero vectors produced 
    # by the upstream embedding pipeline. They carry no signal (cosine 0 with every query) 
    # and would divide by zero below.
    norms = np.linalg.norm(chunk_vectors, axis=1)
    keep = norms > 0
    chunk_vectors = chunk_vectors[keep]
    chunk_to_doc = chunk_to_doc[keep]
    norms = norms[keep]

    # L2-normalize each row so that inner product == cosine similarity.
    # norms[:, None] reshapes from (N,) to (N, 1) so the division broadcasts
    # across the 1024 columns: each row is divided by its own scalar norm.
    chunk_vectors = chunk_vectors / norms[:, None]
    return chunk_vectors, chunk_to_doc


def build_index(chunk_vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Build an exact-cosine FAISS index over already-normalized chunk vectors."""
    dim = chunk_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(chunk_vectors)
    return index
