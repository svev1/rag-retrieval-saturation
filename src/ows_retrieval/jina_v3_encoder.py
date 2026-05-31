"""Jina V3 model loader, used by the dense-embedding investigation probes.

The probes re-encode text under several Jina V3 task modes
(`retrieval.query`, `retrieval.passage`, `text-matching`, `separation`,
and no-task) to figure out which mode the upstream used and whether dense
retrieval works with the shipped vectors. They call `load_jina_v3()` and
then `model.encode(...)` directly, so this module just owns loading the
model once and caching it at module level.
"""
import numpy as np

_MODEL = None


def load_jina_v3():
    """Return the Jina V3 model, loaded once and cached at module level."""
    global _MODEL
    if _MODEL is None:
        from transformers import AutoModel
        _MODEL = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v3",
            trust_remote_code=True,
        ).eval()
    return _MODEL


def encode_query(text: str) -> np.ndarray:
    """Encode a single query into a (1, 1024) float32 L2-normalized vector."""
    model = load_jina_v3()
    vec = model.encode([text], task="retrieval.query")
    vec = np.asarray(vec, dtype=np.float32)
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    return vec
