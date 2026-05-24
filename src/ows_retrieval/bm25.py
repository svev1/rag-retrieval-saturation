"""BM25 indexing and search via rank_bm25, with NLTK English tokenization."""
import re

import nltk
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))


_WORD_RE = re.compile(r"\w+")


def tokenize(text: str) -> list:
    """Lowercase, extract word tokens, drop English stopwords."""
    return [w for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS]


def doc_text(row) -> str:
    """Combine title, description, and main_content for one doc, skipping NaNs."""
    parts = []
    for field in ["title", "description", "main_content"]:
        value = row[field]
        if pd.notna(value):
            parts.append(str(value))
    return " ".join(parts)


def build_index(docs: pd.DataFrame) -> BM25Okapi:
    """Tokenize the docs and build a rank_bm25 BM25Okapi index.

    Position i in the returned index corresponds to docs.iloc[i].
    """
    tokenized_corpus = [tokenize(doc_text(row)) for _, row in docs.iterrows()]
    return BM25Okapi(tokenized_corpus)


def search(retriever: BM25Okapi, query: str, k: int = 10):
    """Run one query against the index, return top-k positions and scores.

    Returns:
        positions: 1D numpy array of k row indices into the docs DataFrame.
        scores: 1D numpy array of k BM25 scores, descending.
    """
    query_tokens = tokenize(query)
    scores = retriever.get_scores(query_tokens)
    top_positions = np.argsort(scores)[::-1][:k]
    return top_positions, scores[top_positions]
