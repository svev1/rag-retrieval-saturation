"""Join RAGAS scores with retrieval noise and summarise the noise-vs-quality story.

Reads scores.parquet (faithfulness + answer_relevancy per answer) and noise.parquet
(retrieval noise per (retriever, topic, k)), joins them on those keys, and writes:
  - summary.csv          mean quality + noise per (retriever, k)
  - quality_vs_k.png     faithfulness and answer relevancy vs k, per retriever
  - noise_vs_k.png       noise (incl. unjudged) and strict noise vs k, per retriever
It also prints the summary table and the row-level Spearman correlation between
noise and each quality metric (NaN faithfulness rows excluded).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; we write PNGs
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

K_VALUES = [1, 3, 5, 10, 20, 50]


def load_joined(scores_path: Path, noise_path: Path) -> pd.DataFrame:
    scores = pd.read_parquet(scores_path)
    noise = pd.read_parquet(noise_path)
    return scores.merge(noise, on=["retriever", "topic_id", "k"], validate="one_to_one")


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean quality and noise per (retriever, k); n_faith counts non-NaN faithfulness."""
    return (
        df.groupby(["retriever", "k"])
        .agg(
            faithfulness=("faithfulness", "mean"),
            answer_relevancy=("answer_relevancy", "mean"),
            noise=("noise", "mean"),
            noise_strict=("noise_strict", "mean"),
            unjudged_rate=("unjudged_rate", "mean"),
            n=("topic_id", "count"),
            n_faith=("faithfulness", "count"),
        )
        .reset_index()
    )


def _line_plot(ax, agg, metric, title):
    for retriever, g in agg.groupby("retriever"):
        ax.plot(g["k"], g[metric], marker="o", label=retriever)
    ax.set_xscale("log")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels(K_VALUES)
    ax.set_xlabel("k")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_quality(agg: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _line_plot(axes[0], agg, "faithfulness", "Faithfulness vs k")
    _line_plot(axes[1], agg, "answer_relevancy", "Answer relevancy vs k")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_noise(agg: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for retriever, g in agg.groupby("retriever"):
        ax.plot(g["k"], g["noise"], marker="o", label=f"{retriever} (incl. unjudged)")
        ax.plot(g["k"], g["noise_strict"], marker="s", linestyle="--", label=f"{retriever} (strict)")
    ax.set_xscale("log")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels(K_VALUES)
    ax.set_xlabel("k")
    ax.set_ylabel("noise")
    ax.set_title("Retrieval noise vs k")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Join scores + noise and summarise RQ3.")
    parser.add_argument("scores", type=Path, help="scores.parquet")
    parser.add_argument("noise", type=Path, help="noise.parquet")
    parser.add_argument("--out-dir", type=Path, default=None, help="where to write outputs (default: scores' dir)")
    args = parser.parse_args()
    out_dir = args.out_dir or args.scores.parent

    df = load_joined(args.scores, args.noise)
    agg = summarise(df)
    agg.to_csv(out_dir / "summary.csv", index=False)
    plot_quality(agg, out_dir / "quality_vs_k.png")
    plot_noise(agg, out_dir / "noise_vs_k.png")

    print(agg.round(3).to_string(index=False))
    print("\nrow-level Spearman correlation (noise vs quality):")
    for metric in ["faithfulness", "answer_relevancy"]:
        sub = df.dropna(subset=[metric])
        rho = sub["noise"].corr(sub[metric], method="spearman")
        print(f"  noise vs {metric:17s} rho = {rho:+.3f}  (n={len(sub)})")
    print(f"\nwrote summary.csv, quality_vs_k.png, noise_vs_k.png to {out_dir}")


if __name__ == "__main__":
    main()
