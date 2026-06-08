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
from ows_retrieval.data import load_docs

# Matches any HTML tag, e.g. <a href="...">, </p>, <h2>. Intentionally simple --
# we are scrubbing noise from scraped pages, not parsing HTML.
_HTML_TAG = re.compile(r"<[^>]+>")

# Repo root: this file is src/ows_retrieval/rag/context.py, so parents[3] is the root.
REPO_ROOT = Path(__file__).resolve().parents[3]

# The frozen run files produced by M1, keyed by retriever name.
RUN_FILES = {
    "bm25": REPO_ROOT / "experiments" / "2026-05-17_bm25_full" / "bm25.run.txt",
    "dense": REPO_ROOT / "experiments" / "2026-05-31_dense_full" / "dense.run.txt",
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


def load_doc_text() -> dict[str, str]:
    """Map every doc_id to its document text, with HTML removed.

    The text is title + description + main_content (via `bm25.doc_text`), then
    stripped of HTML. The retrievers indexed the raw, HTML-laden text, but the
    generator should read clean prose -- so we strip here, on the generation side
    only. The frozen run files (what the retrievers returned) are unaffected.
    """
    docs = load_docs()
    return {row["id"]: strip_html(doc_text(row)) for _, row in docs.iterrows()}


def build_context(
    doc_ids: list[str],
    doc_text_map: dict[str, str],
    char_cap: int = DEFAULT_CHAR_CAP,
) -> str:
    """Turn a ranked list of doc_ids into the labeled context block for the prompt.

    For each doc_id, in rank order: look up its text, collapse runs of whitespace
    (so the cap counts real characters, not newlines), front-truncate to `char_cap`
    (best-ranked content sits at the start of the text), and wrap it in a visible
    `[Document N]` label so the generator can tell the documents apart. The labeled
    snippets are joined with blank lines into one block.

    `doc_ids` is already a top-k slice, so the caller controls how many documents
    land in the block; this function just formats whatever it is given.
    """
    blocks = []
    for n, doc_id in enumerate(doc_ids, start=1):
        text = " ".join(doc_text_map[doc_id].split())
        snippet = text[:char_cap]
        if len(text) > char_cap:
            snippet += "…"
        blocks.append(f"[Document {n}]\n{snippet}")
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
    doc_text_map = load_doc_text()
    check_coverage(doc_text_map)
    print(f"{len(doc_text_map)} documents; every run-file doc_id resolves to text")
