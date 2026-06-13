"""Drive the k-sweep: for each retriever, topic, and k, generate and cache an answer.

This is the orchestrator. It calls the pieces built in the other modules -- it adds
no new retrieval or generation logic of its own:

    read each run file ONCE  ->  ranked doc_ids per topic
    for each (retriever, topic, k):
        skip if already cached            (cache)
        slice the first k doc_ids         (nested prefixes -- no re-search)
        build the labeled context block   (context)
        fill the fixed prompt template    (prompt)
        generate the answer via Ollama    (generator)
        write the provenance record       (cache)

`run_sweep` is the pure loop; `main` reads an experiment config.yaml and calls it,
so the smoke and the full sweeps differ only by their config, never by code.
"""
import argparse
from pathlib import Path

import yaml

from ows_retrieval.data import load_topics
from ows_retrieval.rag.cache import record_path, write_record
from ows_retrieval.rag.context import (
    RUN_FILES,
    build_context,
    load_doc_text,
    ranked_docs_by_topic,
)
from ows_retrieval.rag.generator import gemma_generate
from ows_retrieval.rag.prompt import PROMPT_TEMPLATE_VERSION, build_prompt


def run_sweep(
    *,
    cache_dir: Path,
    retrievers: list[str],
    topic_ids: list[int],
    k_values: list[int],
    model: str,
    options: dict,
    char_cap: int,
    doc_text_source: str = "html",
    think: bool | None = None,
    request_timeout: float = 300.0,
) -> None:
    """Generate and cache one answer per (retriever, topic, k), skipping existing.

    Loads the doc-text map and query texts once, then reads each retriever's run
    file once and slices nested top-k prefixes from it.

    `doc_text_source` picks which text the generator reads (see
    `context.load_doc_text`): "html" (v1) or "clean" (default_text(), pairs with
    the bm25_clean / dense run files). Defaults to "html" so existing configs are
    unchanged.
    """
    queries = load_topics().set_index("topic_id")["query"].to_dict()
    text_map = load_doc_text(doc_text_source)

    generated = skipped = failed = 0
    for retriever in retrievers:
        ranked = ranked_docs_by_topic(RUN_FILES[retriever])
        for topic_id in topic_ids:
            doc_ids_full = ranked[topic_id]
            for k in k_values:
                path = record_path(cache_dir, retriever, topic_id, k)
                if path.exists():
                    skipped += 1
                    print(f"skip  {path.name}")
                    continue

                doc_ids_k = doc_ids_full[:k]
                prompt = build_prompt(
                    queries[topic_id],
                    build_context(doc_ids_k, text_map, char_cap),
                )
                # Tolerate a single failed generation (e.g. transient timeout or
                # GPU OOM) so an unattended sweep keeps going; the uncached key is
                # simply retried on the next run.
                try:
                    result = gemma_generate(
                        prompt, model=model, options=options,
                        timeout=request_timeout, think=think,
                    )
                except RuntimeError as err:
                    failed += 1
                    print(f"FAIL  {path.name}  {err}")
                    continue
                write_record(
                    path,
                    {
                        "retriever": retriever,
                        "topic_id": topic_id,
                        "k": k,
                        "query": queries[topic_id],
                        "doc_ids": doc_ids_k,
                        "prompt": prompt,
                        "answer": result.text,
                        "prompt_tokens": result.prompt_tokens,
                        "output_tokens": result.output_tokens,
                        "model": model,
                        "options": options,
                        "think": think,
                        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                    },
                )
                generated += 1
                print(
                    f"gen   {path.name}  "
                    f"ptok={result.prompt_tokens} otok={result.output_tokens}"
                )

    print(f"\ndone: {generated} generated, {skipped} skipped, {failed} failed, "
          f"{generated + skipped + failed} total")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG k-sweep from a config.")
    parser.add_argument("config", type=Path, help="path to an experiment config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_sweep(
        cache_dir=args.config.parent / "cache",
        retrievers=config["retrievers"],
        topic_ids=config["topic_ids"],
        k_values=config["k_values"],
        model=config["model"],
        options=config["options"],
        char_cap=config["char_cap"],
        doc_text_source=config.get("doc_text_source", "html"),
        think=config.get("think"),
        request_timeout=config.get("request_timeout", 300.0),
    )


if __name__ == "__main__":
    main()
