"""Plot helpers (non-interactive backend so they work from the GUI thread)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_confusion_matrix(cm, title: str, path) -> Path:
    cm = np.asarray(cm)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def plot_metric_bars(groups: dict, metrics: list, title: str, path) -> Path:
    """groups[label][metric] = (mean, std). Grouped bars, one group per metric."""
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(groups)
    x = np.arange(len(metrics))
    w = 0.8 / len(labels)
    for i, lab in enumerate(labels):
        means = [groups[lab][m][0] for m in metrics]
        stds = [groups[lab][m][1] for m in metrics]
        ax.bar(x - 0.4 + w / 2 + i * w, means, w, yerr=stds, capsize=2, label=lab)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score (held-out test set)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_noise_sweep(x_labels: list, series: dict, baselines: dict, path) -> Path:
    """series[label][metric] = (means, stds) aligned with x_labels.
    baselines[label][metric] = value, drawn as dashed horizontal lines."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    xs = np.arange(len(x_labels))
    for ax, metric in zip(axes, ["accuracy", "roc_auc"]):
        for lab, d in series.items():
            ax.errorbar(xs, d[metric][0], yerr=d[metric][1], marker="o", capsize=3, label=lab)
        for lab, d in baselines.items():
            ax.axhline(d[metric], linestyle="--", color="gray", linewidth=1)
            ax.text(xs[-1], d[metric], f" {lab}", va="bottom", ha="right", fontsize=8, color="gray")
        ax.set_xticks(xs, x_labels, rotation=30, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs noise level")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    return _save(fig, path)


def plot_round_curves(curves: dict, path) -> Path:
    """curves[model_label][setting_label][metric] = array over rounds (mean over repeats)."""
    models = list(curves)
    fig, axes = plt.subplots(2, len(models), figsize=(6 * len(models), 7), squeeze=False)
    for j, model in enumerate(models):
        for i, metric in enumerate(["accuracy", "roc_auc"]):
            ax = axes[i][j]
            for setting, d in curves[model].items():
                ax.plot(np.arange(1, len(d[metric]) + 1), d[metric], label=setting)
            ax.set_title(f"{model}: {metric} per round")
            ax.set_xlabel("communication round")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    return _save(fig, path)
