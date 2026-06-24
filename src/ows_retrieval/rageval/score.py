"""Score each row of the eval dataset with RAGAS faithfulness + answer relevancy.

Resumable: one JSON line per (retriever, topic, k) is appended to a .jsonl log and
keys already present are skipped, so a crash is resumed by re-running. Each row is
scored on its own (the local judge runs serially), with judge token usage logged.
NaN scores are stored as null and interpreted later, not hidden:
  - faithfulness is NaN for refusals (no claims to verify) -- expected, reported;
  - a NaN on a non-refusal is a judge glitch -- flagged for re-scoring.

The final parquet plus the printed summary are the inputs to the synthesis step.
"""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd  # imported first: loads pyarrow before langchain_ollama (DLL order)

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.run_config import RunConfig

from ows_retrieval.rageval.judge import make_judge
from ows_retrieval.rageval.tokens import TokenCounter


def _clean(value) -> float | None:
    """NaN -> None so it stores as JSON null rather than an invalid NaN literal."""
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def _done_keys(jsonl_path: Path) -> set:
    """The (retriever, topic_id, k) keys already scored in the resume log."""
    if not jsonl_path.exists():
        return set()
    done = set()
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            done.add((r["retriever"], r["topic_id"], r["k"]))
    return done


def score_dataset(dataset_path: Path, jsonl_path: Path, limit: int | None, timeout: int) -> None:
    """Score every not-yet-done row, appending one JSON line per row as it finishes."""
    df = pd.read_parquet(dataset_path)
    done = _done_keys(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    llm, embeddings = make_judge()
    metrics = [Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=embeddings)]

    todo = [
        row for _, row in df.iterrows()
        if (row.retriever, int(row.topic_id), int(row.k)) not in done
    ]
    if limit is not None:
        todo = todo[:limit]
    print(
        f"{len(done)} already scored; scoring {len(todo)} now; "
        f"{len(df) - len(done) - len(todo)} left after this.",
        flush=True,
    )

    for i, row in enumerate(todo, 1):
        key = (row.retriever, int(row.topic_id), int(row.k))
        sample = SingleTurnSample(
            user_input=row.question,
            response=row.answer,
            retrieved_contexts=[str(c) for c in row.contexts],
        )
        counter = TokenCounter()
        result = evaluate(
            EvaluationDataset(samples=[sample]),
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(timeout=timeout, max_workers=1),
            callbacks=[counter],
            show_progress=False,
        )
        s = result.to_pandas().iloc[0]
        rec = {
            "retriever": row.retriever,
            "topic_id": int(row.topic_id),
            "k": int(row.k),
            "is_refusal": bool(row.is_refusal),
            "faithfulness": _clean(s.get("faithfulness")),
            "answer_relevancy": _clean(s.get("answer_relevancy")),
            "judge_input_tokens": counter.input_tokens,
            "judge_output_tokens": counter.output_tokens,
            "judge_llm_calls": counter.llm_calls,
        }
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(
            f"{datetime.now():%H:%M:%S} [{i}/{len(todo)}] {key}  "
            f"F={rec['faithfulness']}  AR={rec['answer_relevancy']}  "
            f"tok={counter.total_tokens}",
            flush=True,
        )


def build_parquet(jsonl_path: Path, parquet_path: Path) -> pd.DataFrame:
    """Collate the resume log into the final scores table."""
    rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows).sort_values(["retriever", "topic_id", "k"]).reset_index(drop=True)
    df.to_parquet(parquet_path)
    return df


def summarize(df: pd.DataFrame) -> None:
    """Step G: split NaNs (refusal vs glitch) and report means with NaN excluded."""
    nan_f = df["faithfulness"].isna()
    refusal_nan = int((nan_f & df["is_refusal"]).sum())
    glitch_nan = int((nan_f & ~df["is_refusal"]).sum())
    total_tokens = int(df["judge_input_tokens"].sum() + df["judge_output_tokens"].sum())

    print(f"\nscored rows: {len(df)}")
    print(
        f"faithfulness NaN: {int(nan_f.sum())} "
        f"(refusal/undefined: {refusal_nan}, glitch/to-rescore: {glitch_nan})"
    )
    print(f"answer_relevancy NaN: {int(df['answer_relevancy'].isna().sum())}")
    print(f"total judge tokens: {total_tokens:,}")
    if glitch_nan:
        bad = df[nan_f & ~df["is_refusal"]][["retriever", "topic_id", "k"]]
        print("glitch rows to re-score (delete these keys' lines from the .jsonl and re-run):")
        print(bad.to_string(index=False))
    print("\nmean faithfulness by retriever x k (NaN excluded):")
    print(df.pivot_table(index="retriever", columns="k", values="faithfulness").round(2))
    print("\nmean answer_relevancy by retriever x k:")
    print(df.pivot_table(index="retriever", columns="k", values="answer_relevancy").round(2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the eval dataset with RAGAS (resumable).")
    parser.add_argument("dataset", type=Path, help="dataset.parquet from collate")
    parser.add_argument("out", type=Path, help="output scores.parquet (a .jsonl beside it is the resume log)")
    parser.add_argument("--limit", type=int, default=None, help="score at most N new rows (quick test)")
    parser.add_argument("--timeout", type=int, default=900, help="per-row judge timeout in seconds")
    parser.add_argument("--summary-only", action="store_true", help="skip scoring; rebuild parquet + summary")
    args = parser.parse_args()

    jsonl_path = args.out.with_suffix(".jsonl")
    if not args.summary_only:
        score_dataset(args.dataset, jsonl_path, args.limit, args.timeout)
    df = build_parquet(jsonl_path, args.out)
    summarize(df)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
