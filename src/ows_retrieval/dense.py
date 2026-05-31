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


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row to *exactly* unit length.

    The stored vectors are normalized at write time, but float32 rounding
    leaves their norms at 1.0 +/- ~4e-3. IndexFlatIP ranks by inner product,
    and inner product == cosine only when both sides are exactly unit length.
    Dividing each row by its measured norm removes that slop so the scores are
    cosine to float precision (and a vector dotted with itself returns 1.0).

    norms[:, None] reshapes (N,) -> (N, 1) so the division broadcasts across
    every column: each row is divided by its own scalar norm.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    return vectors / norms[:, None]


def search(
    index: faiss.IndexFlatIP,
    doc_ids: np.ndarray,
    query_vector: np.ndarray,
    k: int = 10,
    candidate_depth: int = 1000,
):
    """Run one query against the chunk index, return its top-k DOCUMENTS.

    Returns:
        top_doc_ids: 1D object array of k document ids, best first.
        top_scores:  1D float32 array of their cosine scores, descending.
    """
    # FAISS wants a 2D (n_queries, dim) float32 array; normalize for exact cosine.
    query = normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))

    # Pull a deep pool of chunks: far more than k, since many chunks collapse
    # into the same doc. scores/idxs come back shaped (1, candidate_depth).
    scores, idxs = index.search(query, candidate_depth)
    scores, idxs = scores[0], idxs[0]

    # Max-over-chunks per doc_id: a doc's score is its best-matching chunk's.
    best_score: dict = {}
    for score, idx in zip(scores, idxs):
        doc_id = doc_ids[idx]
        if doc_id not in best_score or score > best_score[doc_id]:
            best_score[doc_id] = float(score)

    # Rank docs by score (descending), keep the top k.
    ranked = sorted(best_score.items(), key=lambda kv: kv[1], reverse=True)[:k]
    top_doc_ids = np.array([doc_id for doc_id, _ in ranked], dtype=object)
    top_scores = np.array([score for _, score in ranked], dtype=np.float32)
    return top_doc_ids, top_scores
