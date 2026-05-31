"""Embedding job orchestration: load dataset, chunk docs, encode, write parquet.

Ported from the standalone owi-embedding tool. ``_embed_docs`` and
``_embed_queries`` are the entry points; everything else (run dirs, parts,
compaction, verification) is resumability and provenance bookkeeping.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import importlib.metadata
import json
import logging
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

from ows_retrieval.embedding.config import Config, load_config
from ows_retrieval.embedding.text import chunk_text, normalize_text, sha256_text

LOGGER = logging.getLogger("ows_retrieval.embedding")
DATASET_LINK_COMMAND = "ln -sfn ~/.owi/public/special/b510baa6-ebd2-11f0-8c43-02a47ca5d9fd ~/.ir_datasets/ows-curlie-2025"


def embed_docs_main() -> None:
    _run(_embed_docs)


def embed_queries_main() -> None:
    _run(_embed_queries)


def verify_main() -> None:
    _run(_verify)


def _run(command: Any) -> None:
    try:
        command()
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _embed_docs() -> None:
    args = _parse_embed_args("Embed OWI document chunks.")
    config = load_config(args.config)
    _setup_logging(config.runtime.log_level)
    dataset = _load_dataset(config)
    docs_iter = _probe_iter(dataset.docs_iter(), config, "docs")
    model = _load_model(config)
    run_dir = _make_run_dir(config, "docs", args.run_id, args.limit_docs)
    metadata = _base_run_metadata(config, "docs", run_dir, args.limit_docs)

    docs_seen = docs_skipped = chunks_written = 0
    part_index = 0
    buffer: list[dict[str, Any]] = []
    progress = tqdm(desc="docs", unit="doc")

    for doc in docs_iter:
        if args.limit_docs is not None and docs_seen >= args.limit_docs:
            break
        docs_seen += 1
        progress.update(1)
        text = normalize_text(_doc_text(doc, config.dataset.source_text_field))
        if not text:
            docs_skipped += 1
            continue
        for chunk in chunk_text(
            text,
            model.tokenizer,
            chunk_size_tokens=config.chunking.chunk_size_tokens,
            chunk_overlap_tokens=config.chunking.chunk_overlap_tokens,
        ):
            buffer.append(
                {
                    "chunk_id": f"{doc.doc_id}:{chunk.chunk_index:06d}",
                    "doc_id": doc.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "token_start": chunk.token_start,
                    "token_end": chunk.token_end,
                    "token_count": chunk.token_count,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "chunk_text_sha256": chunk.text_sha256,
                    "source_dataset": config.dataset.dataset_id,
                    "source_subset": config.dataset.source_subset,
                    "source_text_field": config.dataset.source_text_field,
                    "model_id": config.model.model_id,
                    "model_revision": config.model.model_revision,
                    "embedding_task": config.model.document_task,
                    "prompt_name": config.model.document_prompt_name,
                    "normalized": config.model.normalize_embeddings,
                    "output_dtype": config.model.output_dtype,
                    "_text": chunk.text,
                }
            )
            if len(buffer) >= config.output.write_batch_size:
                chunks_written += _write_doc_part(config, model, run_dir, part_index, buffer)
                part_index += 1
                buffer = []

    if buffer:
        chunks_written += _write_doc_part(config, model, run_dir, part_index, buffer)

    progress.close()
    metadata.update(
        {
            "finished_at": _now(),
            "docs_seen": docs_seen,
            "docs_skipped_empty": docs_skipped,
            "chunks_written": chunks_written,
            "parts_written": len(_part_files(run_dir)),
        }
    )
    _write_metadata(run_dir, metadata)
    final_path = _compact_parts(config, run_dir, config.output.final_doc_artifact, args.limit_docs)
    verify_artifact(final_path, metadata_kind="docs")
    LOGGER.info("Wrote %s", final_path)


def _embed_queries() -> None:
    args = _parse_embed_args("Embed OWI queries.")
    config = load_config(args.config)
    _setup_logging(config.runtime.log_level)
    dataset = _load_dataset(config)
    queries_iter = _probe_iter(dataset.queries_iter(), config, "queries")
    model = _load_model(config)
    run_dir = _make_run_dir(config, "queries", args.run_id, args.limit_docs)
    metadata = _base_run_metadata(config, "queries", run_dir, args.limit_docs)

    rows: list[dict[str, Any]] = []
    for index, query in enumerate(queries_iter):
        if args.limit_docs is not None and index >= args.limit_docs:
            break
        query_id, text, field = _query_text(query)
        text = normalize_text(text)
        if not text:
            continue
        rows.append(
            {
                "query_id": query_id,
                "query_index": index,
                "query_text_sha256": sha256_text(text),
                "source_dataset": config.dataset.dataset_id,
                "source_subset": config.dataset.source_subset,
                "source_text_field": field,
                "model_id": config.model.model_id,
                "model_revision": config.model.model_revision,
                "embedding_task": config.model.query_task,
                "prompt_name": config.model.query_prompt_name,
                "normalized": config.model.normalize_embeddings,
                "output_dtype": config.model.output_dtype,
                "_text": text,
            }
        )

    chunks_written = _write_query_part(config, model, run_dir, 0, rows) if rows else 0
    metadata.update(
        {
            "finished_at": _now(),
            "queries_seen": len(rows),
            "queries_written": chunks_written,
            "parts_written": len(_part_files(run_dir)),
        }
    )
    _write_metadata(run_dir, metadata)
    final_path = _compact_parts(config, run_dir, config.output.final_query_artifact, args.limit_docs)
    verify_artifact(final_path, metadata_kind="queries")
    LOGGER.info("Wrote %s", final_path)


def _verify() -> None:
    parser = argparse.ArgumentParser(description="Verify an embedding Parquet artifact.")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--kind", choices=["docs", "queries"], default="docs")
    args = parser.parse_args()
    verify_artifact(args.artifact, metadata_kind=args.kind)


def verify_artifact(path: Path, *, metadata_kind: str) -> None:
    table = pq.read_table(path)
    if table.num_rows == 0:
        raise RuntimeError(f"{path} has no rows")
    required = {"model_id", "model_revision", "embedding_task", "embedding"}
    missing = required - set(table.column_names)
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
    if metadata_kind == "docs" and "chunk_id" not in table.column_names:
        raise RuntimeError(f"{path} is missing chunk_id")
    if metadata_kind == "queries" and "query_id" not in table.column_names:
        raise RuntimeError(f"{path} is missing query_id")

    embeddings = table.column("embedding")
    first = embeddings[0].as_py()
    if not first:
        raise RuntimeError("first embedding is empty")
    dim = len(first)
    for index in range(table.num_rows):
        value = embeddings[index].as_py()
        if value is None or len(value) != dim:
            raise RuntimeError(f"row {index} has an invalid embedding")

    if "embedding_dim" in table.column_names:
        dims = set(table.column("embedding_dim").to_pylist())
        if dims != {dim}:
            raise RuntimeError(f"embedding_dim column {dims} does not match actual dim {dim}")
    LOGGER.info("Verified %s rows with embedding_dim=%s in %s", table.num_rows, dim, path)


def _parse_embed_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit-docs", type=int, default=None, help="Limit docs or queries for a smoke run.")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def _load_dataset(config: Config) -> Any:
    try:
        import ir_datasets_owi
    except ImportError as exc:
        raise RuntimeError("Install dependencies with `uv sync` before running this command") from exc

    try:
        return ir_datasets_owi.load(config.dataset.dataset_id, batch_size=config.dataset.dataset_batch_size)
    except Exception as exc:
        message = (
            f"Could not load {config.dataset.dataset_id}. Ensure the OWI validation data exists locally "
            "and is linked for ir_datasets, for example:\n"
            f"  {DATASET_LINK_COMMAND}"
        )
        raise RuntimeError(message) from exc


def _probe_iter(items: Any, config: Config, label: str) -> Any:
    iterator = iter(items)
    try:
        first = next(iterator)
    except StopIteration:
        return iter(())
    except Exception as exc:
        message = (
            f"Could not read {label} from {config.dataset.dataset_id}. Ensure the OWI validation data exists locally "
            "and is linked for ir_datasets, for example:\n"
            f"  {DATASET_LINK_COMMAND}"
        )
        raise RuntimeError(message) from exc

    def replay() -> Any:
        yield first
        yield from iterator

    return replay()


def _load_model(config: Config) -> SentenceTransformer:
    device = _select_device(config.model.device)
    LOGGER.info("Loading %s at %s on %s", config.model.model_id, config.model.model_revision, device)
    return SentenceTransformer(
        config.model.model_id,
        revision=config.model.model_revision,
        trust_remote_code=config.model.trust_remote_code,
        device=device,
    )


def _select_device(configured: str) -> str:
    if configured != "auto":
        return configured
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _write_doc_part(
    config: Config,
    model: SentenceTransformer,
    run_dir: Path,
    part_index: int,
    rows: list[dict[str, Any]],
) -> int:
    return _write_part(
        config,
        model,
        run_dir,
        part_index,
        rows,
        config.model.document_task,
        config.model.document_prompt_name,
    )


def _write_query_part(
    config: Config,
    model: SentenceTransformer,
    run_dir: Path,
    part_index: int,
    rows: list[dict[str, Any]],
) -> int:
    return _write_part(
        config,
        model,
        run_dir,
        part_index,
        rows,
        config.model.query_task,
        config.model.query_prompt_name,
    )


def _write_part(
    config: Config,
    model: SentenceTransformer,
    run_dir: Path,
    part_index: int,
    rows: list[dict[str, Any]],
    task: str,
    prompt_name: str | None,
) -> int:
    parts_dir = run_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    path = parts_dir / f"part-{part_index:06d}.parquet"
    if path.exists():
        LOGGER.info("Skipping existing %s", path)
        return pq.read_metadata(path).num_rows

    texts = [row.pop("_text") for row in rows]
    encode_kwargs = {
        "batch_size": config.runtime.encode_batch_size,
        "normalize_embeddings": config.model.normalize_embeddings,
        "show_progress_bar": False,
    }
    if task:
        encode_kwargs["task"] = task
    if prompt_name:
        encode_kwargs["prompt_name"] = prompt_name
    embeddings = model.encode(texts, **encode_kwargs)
    embeddings = np.asarray(embeddings, dtype=config.model.output_dtype)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
        raise RuntimeError("Model returned an unexpected embedding shape")
    if np.isnan(embeddings).any():
        raise RuntimeError("Model returned NaN embeddings")

    for row, embedding in zip(rows, embeddings, strict=True):
        row["embedding_dim"] = int(embeddings.shape[1])
        row["embedding"] = embedding.astype(config.model.output_dtype).tolist()

    table = pa.Table.from_pylist(rows, schema=_schema(rows[0], config.model.output_dtype))
    pq.write_table(table, path, compression="zstd")
    LOGGER.info("Wrote %s rows to %s", len(rows), path)
    return len(rows)


def _schema(sample: dict[str, Any], output_dtype: str) -> pa.Schema:
    fields = []
    float_type = pa.float32() if output_dtype == "float32" else pa.float16()
    for key, value in sample.items():
        if key == "embedding":
            fields.append(pa.field("embedding", pa.list_(float_type)))
        elif isinstance(value, bool):
            fields.append(pa.field(key, pa.bool_()))
        elif isinstance(value, int):
            fields.append(pa.field(key, pa.int64()))
        else:
            fields.append(pa.field(key, pa.string()))
    return pa.schema(fields)


def _compact_parts(config: Config, run_dir: Path, artifact_name: str, limit_docs: int | None) -> Path:
    part_files = _part_files(run_dir)
    if not part_files:
        raise RuntimeError(f"No part files found in {run_dir / 'parts'}")
    suffix = ".smoke.parquet" if limit_docs is not None else ".parquet"
    final_name = artifact_name.replace(".parquet", suffix)
    final_path = config.output.embeddings_dir / final_name
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tables = [pq.read_table(path) for path in part_files]
    table = pa.concat_tables(tables, promote_options="default")
    if "chunk_id" in table.column_names:
        chunk_ids = table.column("chunk_id").to_pylist()
        if len(chunk_ids) != len(set(chunk_ids)):
            raise RuntimeError("Duplicate chunk_id values found")
    pq.write_table(table, final_path, compression="zstd")
    return final_path


def _part_files(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "parts").glob("part-*.parquet"))


def _doc_text(doc: Any, field: str) -> str:
    value = getattr(doc, field)
    if callable(value):
        value = value()
    return value or ""


def _query_text(query: Any) -> tuple[str, str, str]:
    query_id = getattr(query, "query_id", None) or getattr(query, "qid", None)
    if query_id is None:
        raise RuntimeError(f"Could not find query id on {query!r}")
    for field in ("default_text", "text", "title", "query"):
        if hasattr(query, field):
            value = getattr(query, field)
            if callable(value):
                value = value()
            if value:
                return str(query_id), str(value), field
    raise RuntimeError(f"Could not find query text on {query!r}")


def _make_run_dir(config: Config, kind: str, run_id: str | None, limit_docs: int | None) -> Path:
    smoke = "smoke-" if limit_docs is not None else ""
    name = run_id or f"{smoke}{kind}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_dir = config.output.runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _base_run_metadata(config: Config, kind: str, run_dir: Path, limit_docs: int | None) -> dict[str, Any]:
    return {
        "kind": kind,
        "run_dir": str(run_dir),
        "started_at": _now(),
        "limit_docs": limit_docs,
        "config_path": str(config.path),
        "config": _jsonable(asdict(config)),
        "versions": _versions(),
        "device": _select_device(config.model.device),
    }


def _write_metadata(run_dir: Path, metadata: dict[str, Any]) -> None:
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _versions() -> dict[str, str]:
    packages = ["torch", "sentence-transformers", "transformers", "pyarrow", "ir-datasets", "ir-datasets-owi"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=level.upper(), format="%(levelname)s %(message)s")


def _now() -> str:
    return datetime.now(UTC).isoformat()
