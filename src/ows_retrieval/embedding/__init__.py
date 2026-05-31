"""Local re-embedding pipeline for the radboud-val split.

Ported into this project so the Jina V5 nano embedding artifacts in
``data/raw/`` are reproducible from source. The pipeline loads the
``ows-curlie-2025/radboud-val`` split, chunks each document with the model
tokenizer (1024-token windows, 128 overlap), and encodes chunks and queries
with the asymmetric ``document`` / ``query`` prompts.

This subpackage has heavy dependencies (torch, sentence-transformers,
ir-datasets-owi) that the core BM25 / FAISS baseline does not need. They live
in the ``embedding`` optional-dependency group; install with
``pip install -e ".[embedding]"`` before running the CLI.
"""
__all__ = ["__version__"]

__version__ = "0.1.0"
