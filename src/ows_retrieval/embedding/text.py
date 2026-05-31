"""Text normalization and tokenizer-based chunking.

Pure functions with no model or dataset dependency. ``chunk_text`` slides a
fixed-size window over the document's token ids and records, for each chunk,
the token span, char span, and a sha256 of the chunk text — enough provenance
to recover the exact text later without storing it in the artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable


@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    text_sha256: str
    text: str

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(
    text: str,
    tokenizer: Any,
    *,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> list[Chunk]:
    if not text:
        return []

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise RuntimeError("Tokenizer must provide offset mappings for traceable chunks")
    if len(input_ids) != len(offsets):
        raise RuntimeError("Tokenizer returned mismatched token and offset counts")
    if not input_ids:
        return []

    chunks: list[Chunk] = []
    step = chunk_size_tokens - chunk_overlap_tokens
    token_start = 0
    chunk_index = 0
    while token_start < len(input_ids):
        token_end = min(token_start + chunk_size_tokens, len(input_ids))
        char_start, char_end = _char_span(offsets[token_start:token_end])
        chunk = text[char_start:char_end]
        chunks.append(
            Chunk(
                chunk_index=chunk_index,
                token_start=token_start,
                token_end=token_end,
                char_start=char_start,
                char_end=char_end,
                text_sha256=sha256_text(chunk),
                text=chunk,
            )
        )
        if token_end == len(input_ids):
            break
        token_start += step
        chunk_index += 1

    return chunks


def _char_span(offsets: Iterable[tuple[int, int]]) -> tuple[int, int]:
    non_empty = [(start, end) for start, end in offsets if end > start]
    if not non_empty:
        return 0, 0
    return non_empty[0][0], non_empty[-1][1]
