"""Compare the manual audit ratings against the RAGAS scores to gauge agreement.

Reads the manual ratings from the filled audit sheet (audit_sheet.xlsx) and joins
them with scores.parquet on (retriever, topic_id, k) -- this is the only step where
the blind ratings meet the RAGAS scores. Maps the manual labels to an ordinal scale
(No=0, Partly=0.5, Yes=1) and reports whether RAGAS tracks the human judgment: the
mean RAGAS score per manual category, the Spearman correlation where the labels vary,
and the row-level comparison (audit_agreement.csv).
"""
import argparse
from pathlib import Path

import pandas as pd

ORDINAL = {"No": 0.0, "Partly": 0.5, "Yes": 1.0}


def load_joined(sheet_xlsx: Path, scores_path: Path) -> pd.DataFrame:
    manual = pd.read_excel(sheet_xlsx)[
        ["audit_id", "retriever", "topic_id", "k", "manual_faithfulness", "manual_relevancy"]
    ]
    scores = pd.read_parquet(scores_path)[
        ["retriever", "topic_id", "k", "faithfulness", "answer_relevancy"]
    ]
    return manual.merge(scores, on=["retriever", "topic_id", "k"], how="left", validate="one_to_one")


def _report(df: pd.DataFrame, manual_col: str, ragas_col: str, name: str) -> None:
    print(f"\n=== {name} ===")
    print(f"manual labels: {df[manual_col].value_counts().to_dict()}")
    by_label = df.groupby(manual_col)[ragas_col].agg(["mean", "min", "max", "count"]).round(3)
    print("mean RAGAS score by manual label:")
    print(by_label.to_string())
    ordinal = df[manual_col].map(ORDINAL)
    if ordinal.nunique() > 1 and df[ragas_col].notna().sum() > 1:
        rho = ordinal.corr(df[ragas_col], method="spearman")
        print(f"Spearman(manual, RAGAS) = {rho:+.3f}")
    else:
        print("Spearman: n/a (manual labels do not vary)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure manual-audit vs RAGAS agreement.")
    parser.add_argument("sheet", type=Path, help="audit_sheet.xlsx (filled manual ratings)")
    parser.add_argument("scores", type=Path, help="scores.parquet (RAGAS scores)")
    parser.add_argument("--out", type=Path, default=None, help="joined comparison CSV (default: beside the sheet)")
    args = parser.parse_args()

    df = load_joined(args.sheet, args.scores).sort_values("audit_id")
    cols = [
        "audit_id", "retriever", "topic_id", "k",
        "manual_faithfulness", "faithfulness", "manual_relevancy", "answer_relevancy",
    ]
    print(df[cols].round(3).to_string(index=False))
    _report(df, "manual_faithfulness", "faithfulness", "FAITHFULNESS")
    _report(df, "manual_relevancy", "answer_relevancy", "ANSWER RELEVANCY")

    out = args.out or args.sheet.with_name("audit_agreement.csv")
    df[cols].to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
