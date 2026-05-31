"""Typed configuration for the embedding job, loaded from a TOML file.

Frozen dataclasses mirror the sections of ``configs/radboud-val.toml``. This
module holds no logic beyond loading, path resolution, and validation — it is
the recipe that determines the artifact (model + revision, prompts, chunking,
dtype, normalization).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class DatasetConfig:
    collection_id: str
    dataset_id: str
    source_subset: str
    source_text_field: str
    dataset_batch_size: int


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    model_revision: str
    document_task: str
    query_task: str
    document_prompt_name: str | None
    query_prompt_name: str | None
    trust_remote_code: bool
    normalize_embeddings: bool
    output_dtype: str
    device: str


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size_tokens: int
    chunk_overlap_tokens: int


@dataclass(frozen=True)
class OutputConfig:
    runs_dir: Path
    embeddings_dir: Path
    write_batch_size: int
    final_doc_artifact: str
    final_query_artifact: str


@dataclass(frozen=True)
class RuntimeConfig:
    encode_batch_size: int
    log_level: str


@dataclass(frozen=True)
class Config:
    path: Path
    dataset: DatasetConfig
    model: ModelConfig
    chunking: ChunkingConfig
    output: OutputConfig
    runtime: RuntimeConfig


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    base_dir = config_path.parent.parent
    dataset = DatasetConfig(**raw["dataset"])
    model = ModelConfig(**raw["model"])
    chunking = ChunkingConfig(**raw["chunking"])

    output_raw = raw["output"].copy()
    output_raw["runs_dir"] = _resolve_path(base_dir, output_raw["runs_dir"])
    output_raw["embeddings_dir"] = _resolve_path(base_dir, output_raw["embeddings_dir"])
    output = OutputConfig(**output_raw)

    runtime = RuntimeConfig(**raw["runtime"])
    config = Config(config_path, dataset, model, chunking, output, runtime)
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    if config.chunking.chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if config.chunking.chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens cannot be negative")
    if config.chunking.chunk_overlap_tokens >= config.chunking.chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
    if config.runtime.encode_batch_size <= 0:
        raise ValueError("encode_batch_size must be positive")
    if config.output.write_batch_size <= 0:
        raise ValueError("write_batch_size must be positive")
    if config.model.output_dtype not in {"float32", "float16"}:
        raise ValueError("output_dtype must be float32 or float16")
    if config.model.model_revision in {"", "main"}:
        raise ValueError("model_revision must be pinned to a commit SHA, not main")


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
