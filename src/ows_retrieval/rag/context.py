"""Resolve retrieved documents into the text context fed to the generator.

Each retriever saved a TREC run file listing the top doc_ids per topic. Those ids
are opaque hashes, but the generator needs real text. This module reads a run file
into ranked id lists, maps each id to its document text, and checks that every id
resolves to a document.
"""
import html
import re
from pathlib import Path

from ows_retrieval.bm25 import doc_text
from ows_retrieval.data import load_docs, load_owi_docs

# Matches any HTML tag, e.g. <a href="...">, </p>, <h2>. Intentionally simple --
# we are scrubbing noise from scraped pages, not parsing HTML.
_HTML_TAG = re.compile(r"<[^>]+>")

# Repo root: this file is src/ows_retrieval/rag/context.py, so parents[3] is the root.
REPO_ROOT = Path(__file__).resolve().parents[3]

# The frozen run files produced by M1, keyed by retriever name. `bm25_clean` is
# the v2 index built over default_text() (matches the dense text source).
RUN_FILES = {
    "bm25": REPO_ROOT / "experiments" / "2026-05-17_bm25_full" / "bm25.run.txt",
    "dense": REPO_ROOT / "experiments" / "2026-05-31_dense_full" / "dense.run.txt",
    "bm25_clean": REPO_ROOT / "experiments" / "2026-06-09_bm25_clean_full" / "bm25_clean.run.txt",
}

# Per-document character budget. Each retrieved doc is front-truncated to this many
# characters so that even the deepest sweep (k=50 docs) fits inside the generator's
# context window. Sized against Gemma 2's 8192-token window (see build_context).
DEFAULT_CHAR_CAP = 500


def ranked_docs_by_topic(run_path: Path) -> dict[int, list[str]]:
    """Read a TREC run file into {topic_id: [doc_id, ...]}, ranked best first.

    Does three things: parse each line, group the lines by topic_id, then sort
    each topic's docs by rank (keeping only the doc_ids, best first). A run line
    looks like: `topic_id Q0 doc_id rank score tag`.
    """
    ranked: dict[int, list[tuple[int, str]]] = {}
    for line in Path(run_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        topic_id, _, doc_id, rank, _, _ = line.split()
        ranked.setdefault(int(topic_id), []).append((int(rank), doc_id))

    return {
        topic_id: [doc_id for _, doc_id in sorted(docs)]
        for topic_id, docs in ranked.items()
    }


def strip_html(text: str) -> str:
    """Remove HTML tags and unescape entities (e.g. &amp; -> &) from doc text.

    Whitespace is left alone here; `build_context` collapses it when truncating.
    """
    return html.unescape(_HTML_TAG.sub(" ", text))


def load_doc_text(source: str = "html") -> dict[str, str]:
    """Map every doc_id to its document text, keyed by `source`.

    Two sources, matching the two index variants:

    - "html" (default, v1): title + description + main_content (via
      `bm25.doc_text`), then run through the crude `strip_html`. This pairs with
      the original HTML BM25 index.
    - "clean" (v2): `default_text()` (title + SAX-stripped main_content) straight
      from `load_owi_docs` -- no `strip_html`, no `description`. This is the exact
      text the dense and bm25_clean retrievers ranked on, so the generator reads
      the same prose it retrieved.

    Either way the return shape is {doc_id: text}; `build_context` is unaffected.
    """
    if source == "clean":
        docs = load_owi_docs()
        return dict(zip(docs["id"], docs["default_text"]))
    if source != "html":
        raise ValueError(f"unknown doc text source: {source!r}")
    docs = load_docs()
    return {row["id"]: strip_html(doc_text(row)) for _, row in docs.iterrows()}


def shape_document(text: str, char_cap: int = DEFAULT_CHAR_CAP) -> str:
    """Collapse runs of whitespace and front-truncate one document to `char_cap`.

    Whitespace is collapsed so the cap counts real characters, not newlines; the
    best-ranked content sits at the start, so a front truncation keeps it.
    """
    collapsed = " ".join(text.split())
    snippet = collapsed[:char_cap]
    if len(collapsed) > char_cap:
        snippet += "…"
    return snippet


def build_context(
    doc_ids: list[str],
    doc_text_map: dict[str, str],
    char_cap: int = DEFAULT_CHAR_CAP,
) -> str:
    """Turn a ranked list of doc_ids into the labeled context block for the prompt.

    Each doc, in rank order, is shaped by `shape_document` and wrapped in a visible
    `[Document N]` label so the generator can tell them apart; the snippets are
    joined with blank lines. `doc_ids` is already a top-k slice.
    """
    blocks = [
        f"[Document {n}]\n{shape_document(doc_text_map[doc_id], char_cap)}"
        for n, doc_id in enumerate(doc_ids, start=1)
    ]
    return "\n\n".join(blocks)


def check_coverage(doc_text_map: dict[str, str]) -> None:
    """Raise if any doc_id in a run file is missing from `doc_text_map`.

    A missing id would silently feed the generator an empty document, so we fail
    loudly here instead.
    """
    for retriever, run_path in RUN_FILES.items():
        run = ranked_docs_by_topic(run_path)
        doc_ids = {doc_id for docs in run.values() for doc_id in docs}
        missing = doc_ids - doc_text_map.keys()
        assert not missing, f"{retriever}: {len(missing)} doc_ids have no text"


if __name__ == "__main__":
    for source in ("html", "clean"):
        doc_text_map = load_doc_text(source)
        check_coverage(doc_text_map)
        print(f"{source:5s}: {len(doc_text_map)} documents; every run-file doc_id resolves to text")
