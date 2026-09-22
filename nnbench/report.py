"""Build comparison plots and a Markdown report from a finished run directory.

Usage: python -m nnbench.report results/<run_dir>
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .metrics import mcnemar_p  # noqa: E402

# Categorical slots, assigned in fixed order to architectures (never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "semibold", "axes.titlesize": 11, "axes.titlelocation": "left",
    "font.size": 9.5, "legend.frameon": False, "lines.linewidth": 2,
})

CURVE_METRICS = [
    ("test_accuracy", "Test accuracy", "higher is better"),
    ("test_macro_f1", "Test macro-F1", "higher is better"),
    ("test_nll", "Test loss (NLL)", "lower is better"),
    ("test_ece", "Test calibration error (ECE)", "lower is better"),
]


def load_run(run_dir):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    finals = [json.loads(p.read_text()) for p in sorted((run_dir / "final").glob("*.json"))]
    history = pd.concat([pd.read_csv(p) for p in sorted((run_dir / "history").glob("*.csv"))],
                        ignore_index=True)
    order = [a for a in cfg["architectures"] if any(f["architecture"] == a for f in finals)]
    return cfg, finals, history, order


def summarise(finals, order):
    df = pd.DataFrame([{k: v for k, v in f.items()
                        if k not in ("per_class_f1", "confusion_matrix")} for f in finals])
    cols = ["params", "epochs_trained", "best_epoch", "train_time_s", "val_accuracy",
            "test_accuracy", "test_accuracy_ci_low", "test_accuracy_ci_high", "test_top3_accuracy",
            "test_macro_precision", "test_macro_recall", "test_macro_f1", "test_nll", "test_ece",
            "throughput_img_per_s", "latency_ms_single"]
    agg = df.groupby("architecture")[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg.insert(0, "n_seeds", df.groupby("architecture").size())
    return df, agg.loc[order]


def _colors(order):
    return {a: SERIES[i % len(SERIES)] for i, a in enumerate(order)}


def plot_test_curves(history, order, out):
    colors = _colors(order)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, (col, title, hint) in zip(axes.flat, CURVE_METRICS):
        if col not in history:
            continue
        for arch in order:
            h = history[history.architecture == arch].dropna(subset=[col])
            m = h.groupby("epoch")[col].agg(["mean", "min", "max"])
            ax.plot(m.index, m["mean"], color=colors[arch], label=arch)
            if h.seed.nunique() > 1:
                ax.fill_between(m.index, m["min"], m["max"], color=colors[arch],
                                alpha=0.12, linewidth=0)
        ax.set_title(f"{title}  ({hint})")
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(order), bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Independent test-set metrics during training (mean over seeds; band = range)",
                 y=1.04, fontsize=12, fontweight="semibold", x=0.06, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_generalisation_gap(history, order, out):
    colors = _colors(order)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arch in order:
        h = history[history.architecture == arch].dropna(subset=["test_accuracy"])
        gap = (h["accuracy"] - h["test_accuracy"]).groupby(h["epoch"]).mean()
        ax.plot(gap.index, gap.values, color=colors[arch], label=arch)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_title("Generalisation gap: train accuracy − test accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy gap")
    ax.legend(loc="upper left", ncol=min(len(order), 3))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_final_accuracy(agg, order, out):
    colors = _colors(order)
    s = agg.sort_values("test_accuracy_mean")
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(s) + 1.4))
    y = np.arange(len(s))
    mean = s["test_accuracy_mean"].values
    err = np.vstack([mean - s["test_accuracy_ci_low_mean"].values,
                     s["test_accuracy_ci_high_mean"].values - mean])
    ax.barh(y, mean, height=0.6, color=[colors[a] for a in s.index])
    ax.errorbar(mean, y, xerr=err, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
    for yi, m in zip(y, mean):
        ax.text(m + err[1].max() + 0.004, yi, f"{m:.3f}", va="center", color=INK)
    ax.set_yticks(y, s.index)
    ax.set_xlim(max(0.0, mean.min() - 0.1), min(1.0, mean.max() + 0.05))
    ax.grid(axis="y", visible=False)
    ax.set_title("Final test accuracy (best-on-validation checkpoint, 95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_efficiency(agg, order, out):
    colors = _colors(order)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    panels = [("params_mean", "Parameters (log scale)", True),
              ("train_time_s_mean", "Training wall time, s (log scale)", True),
              ("latency_ms_single_mean", "Single-image latency, ms", False)]
    for ax, (col, label, log) in zip(axes, panels):
        for arch in order:
            x, yv = agg.loc[arch, col], agg.loc[arch, "test_accuracy_mean"]
            ax.scatter(x, yv, s=70, color=colors[arch], edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(arch, (x, yv), textcoords="offset points", xytext=(7, 4), color=INK,
                        fontsize=8.5)
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(label)
    axes[0].set_ylabel("test accuracy")
    fig.suptitle("Accuracy vs. cost", x=0.02, ha="left", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_per_class_f1(finals, order, class_names, out):
    rows = [np.mean([f["per_class_f1"] for f in finals if f["architecture"] == a], axis=0)
            for a in order]
    data = np.array(rows)
    fig, ax = plt.subplots(figsize=(1.0 * len(class_names) + 2, 0.55 * len(order) + 1.6))
    im = ax.imshow(data, cmap=SEQ_BLUE, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    ax.set_yticks(range(len(order)), order)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(data):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="#ffffff" if v > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title("Per-class test F1")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_confusion(final, class_names, out):
    cm = np.array(final["confusion_matrix"], dtype=float)
    cm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap=SEQ_BLUE, vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    for (i, j), v in np.ndenumerate(cm):
        if v >= 0.01:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if v > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Row-normalised test confusion matrix — {final['architecture']} "
                 f"(seed {final['seed']})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def mcnemar_table(run_dir, order, seed):
    y = np.load(run_dir / "predictions" / "y_test.npy")
    correct = {a: np.load(run_dir / "predictions" / f"{a}_seed{seed}.npy").argmax(1) == y
               for a in order}
    table = pd.DataFrame(index=order, columns=order, dtype=object)
    for a, b in itertools.product(order, order):
        table.loc[a, b] = "—" if a == b else f"{mcnemar_p(correct[a], correct[b]):.2g}"
    return table


def _fmt(mean, std, digits=4):
    return f"{mean:.{digits}f}" if np.isnan(std) else f"{mean:.{digits}f} ± {std:.{digits}f}"


def write_report(run_dir):
    run_dir = Path(run_dir)
    cfg, finals, history, order = load_run(run_dir)
    per_run, agg = summarise(finals, order)
    per_run.to_csv(run_dir / "results_per_run.csv", index=False)
    agg.to_csv(run_dir / "summary.csv")

    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    class_names = cfg["class_names"]
    plot_test_curves(history, order, plots / "test_curves.png")
    plot_generalisation_gap(history, order, plots / "generalisation_gap.png")
    plot_final_accuracy(agg, order, plots / "final_accuracy.png")
    plot_efficiency(agg, order, plots / "efficiency.png")
    plot_per_class_f1(finals, order, class_names, plots / "per_class_f1.png")
    best_arch = agg["test_accuracy_mean"].idxmax()
    best_run = max((f for f in finals if f["architecture"] == best_arch),
                   key=lambda f: f["test_accuracy"])
    plot_confusion(best_run, class_names, plots / "confusion_best.png")

    ranked = agg.sort_values("test_accuracy_mean", ascending=False)
    lines = [
        f"# Architecture comparison — {cfg['dataset']}",
        "",
        f"Seeds: {cfg['seeds']} · max epochs: {cfg['epochs']} · batch size: {cfg['batch_size']} · "
        f"train/val/test: {cfg['n_train']}/{cfg['n_val']}/{cfg['n_test']}",
        "",
        "The test set is the dataset's official held-out split. It is never used for training, "
        "normalisation statistics, early stopping or checkpoint selection; it is only *observed* "
        "every epoch so we can see how each architecture's generalisation evolves.",
        "",
        "## Final test metrics (checkpoint selected on validation)",
        "",
        "| Rank | Architecture | Params | Test acc (95% CI) | Macro-F1 | Top-3 | NLL | ECE | "
        "Best epoch | Train time (s) | Latency (ms) |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (arch, r) in enumerate(ranked.iterrows(), 1):
        lines.append(
            f"| {rank} | **{arch}** | {int(r.params_mean):,} | "
            f"{_fmt(r.test_accuracy_mean, r.test_accuracy_std)} "
            f"[{r.test_accuracy_ci_low_mean:.3f}, {r.test_accuracy_ci_high_mean:.3f}] | "
            f"{_fmt(r.test_macro_f1_mean, r.test_macro_f1_std)} | {r.test_top3_accuracy_mean:.4f} | "
            f"{r.test_nll_mean:.3f} | {r.test_ece_mean:.3f} | {r.best_epoch_mean:.0f} | "
            f"{r.train_time_s_mean:.0f} | {r.latency_ms_single_mean:.2f} |")
    seed0 = cfg["seeds"][0]
    lines += [
        "",
        "## Pairwise significance (exact McNemar test, p-values, seed "
        f"{seed0})",
        "",
        "Small p (< 0.05) means the two models' test errors differ by more than chance.",
        "",
        mcnemar_table(run_dir, order, seed0).to_markdown(),
        "",
        "## Test metrics across training",
        "",
        "![test curves](plots/test_curves.png)",
        "",
        "![generalisation gap](plots/generalisation_gap.png)",
        "",
        "## Final accuracy and cost",
        "",
        "![final accuracy](plots/final_accuracy.png)",
        "",
        "![efficiency](plots/efficiency.png)",
        "",
        "## Per-class behaviour",
        "",
        "![per-class F1](plots/per_class_f1.png)",
        "",
        "![confusion](plots/confusion_best.png)",
        "",
    ]
    (run_dir / "report.md").write_text("\n".join(lines))
    print(f"report written to {run_dir / 'report.md'}")
    print(ranked[["test_accuracy_mean", "test_macro_f1_mean", "test_ece_mean", "params_mean"]]
          .to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    write_report(p.parse_args().run_dir)


if __name__ == "__main__":
    main()
