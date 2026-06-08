"""Read and write the per-answer cache for the k-sweep.

Each generated answer is stored as one JSON file named by its key
(retriever, topic_id, k), so "do we already have this answer?" is just
"does this file exist?". That makes the sweep resumable: a crashed run is
restarted by re-running it, and existing files are skipped.

This module is pure file I/O. The sweep decides *when* to skip and assembles
each record dict; the cache only builds paths and reads/writes files.
"""
import json
from pathlib import Path


def record_path(cache_dir: Path, retriever: str, topic_id: int, k: int) -> Path:
    """Build the cache filename for one (retriever, topic, k) answer.

    One subfolder per topic keeps the cache navigable at full scale (28 topics ×
    2 retrievers × 6 k = 336 files): cache_dir / "t24" / "bm25__k3.json". The path
    IS the key, so existence of the file answers "is this already cached?".
    """
    return Path(cache_dir) / f"t{topic_id}" / f"{retriever}__k{k}.json"


def write_record(path: Path, record: dict) -> None:
    """Write one answer record to disk as readable JSON.

    Creates the cache directory if needed. ensure_ascii=False keeps non-ASCII text
    (e.g. Icelandic Þ, ö) human-readable instead of \\uXXXX escapes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def read_record(path: Path) -> dict:
    """Load one answer record back from disk (used at collation time, Step 7)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
