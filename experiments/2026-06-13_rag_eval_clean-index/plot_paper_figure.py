"""Render the paper's Figure 2 (quality_vs_k_paper.png) from summary.csv.

Faithfulness and answer relevancy vs k, one line per retriever. Faithfulness reflects
the refusal masking applied in synthesis.load_joined (refusals -> NaN, excluded from the
mean). Run from the experiment directory:  python3 plot_paper_figure.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
K_VALUES = [1, 3, 5, 10, 20, 50]
STYLE = {"bm25_clean": ("BM25", "#1f77b4"), "dense": ("Dense", "#d62728")}

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13})

agg = pd.read_csv(HERE / "summary.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
panels = [("faithfulness", "faithfulness", "Faithfulness vs k", "lower left"),
          ("answer_relevancy", "answer relevancy", "Answer relevancy vs k", "upper right")]
for ax, (col, ylab, title, legloc) in zip(axes, panels):
    for retr, g in agg.groupby("retriever"):
        label, color = STYLE.get(retr, (retr, None))
        ax.plot(g["k"], g[col], marker="o", markersize=8, linewidth=2, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels(K_VALUES)
    ax.set_xlabel("k (retrieved documents)")
    ax.set_ylabel(ylab)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc=legloc)
fig.tight_layout()
fig.savefig(HERE / "quality_vs_k_paper.png", dpi=150)
plt.close(fig)
print("wrote quality_vs_k_paper.png")
