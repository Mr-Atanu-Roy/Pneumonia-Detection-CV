from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb
from sklearn.metrics import auc


def plot_and_log_curves(
    cm: List[List[int]],
    pr_curve: Dict[str, Any],
    roc_curve: Dict[str, Any],
    class_names: List[str],
    figsize: Tuple[int, int] = (22, 6),
    active_run: Optional[wandb.sdk.wandb_run.Run] = None,
):
    """
    Plots Confusion Matrix, Precision-Recall curve, and ROC curve in a single
    row and displays them in the notebook. Optionally logs all
    three charts to the active W&B run.

    Args:
        - cm (List[List[int]]): 2×2 confusion matrix as nested lists [[TN, FP], [FN, TP]].
        - pr_curve (Dict): Dict with keys "precision", "recall", "thresholds" as returned by EvaluationMetrics.compute().
        - roc_curve (Dict): Dict with keys "fpr", "tpr", "thresholds" as returned by EvaluationMetrics.compute().
        - class_names (List[str]): Display labels (e.g. ["Normal", "Pneumonia"]).
        - figsize (Tuple[int, int]): Figure size in inches.
                                     Defaults to (22, 6).
        - active_run Optional[wandb.sdk.wandb_run.Run]: Optional active W&B run to log charts to. If None, charts will not be logged to W&B.
                                                        Defaults to None.

    Raises:
        - ValueError: If pr_curve or roc_curve dicts are missing required keys.
    """

    # Validate weather given pr_curve and roc_curve dicts have the required keys
    required_pr_keys = {"precision", "recall", "threshold"}
    required_roc_keys = {"fpr", "tpr", "threshold"}
    if not required_pr_keys.issubset(pr_curve.keys()):
        raise ValueError(
            f"pr_curve dict is missing required keys. Required keys: {required_pr_keys}. Given keys: {pr_curve.keys()}"
        )

    if not required_roc_keys.issubset(roc_curve.keys()):
        raise ValueError(
            f"roc_curve dict is missing required keys. Required keys: {required_roc_keys}. Given keys: {roc_curve.keys()}"
        )

    print("\n[INFO] Plotting Confusion Matrix, PR Curve, and ROC Curve...\n")

    recall = pr_curve["recall"]
    precision = pr_curve["precision"]
    fpr = roc_curve["fpr"]
    tpr = roc_curve["tpr"]

    group_names = ["TN", "FP", "FN", "TP"]

    pr_auc = auc(recall, precision)
    roc_auc = auc(fpr, tpr)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=120)

    # 1. Confusion Matrix ----------------------------

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=2,
        linecolor="white",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 16},
        ax=axes[0],
    )

    axes[0].set_title("Confusion Matrix", fontsize=18, weight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=12, weight="bold")
    axes[0].set_ylabel("True Label", fontsize=12, weight="bold")

    # Add TN, FP, FN, TP labels
    for text, label in zip(axes[0].texts, group_names):
        text.set_text(f"{text.get_text()}\n({label})")

    # 2. PR Curve ----------------------------

    axes[1].plot(recall, precision, linewidth=2.5, label=f"PR AUC = {pr_auc:.4f}")

    axes[1].fill_between(recall, precision, alpha=0.2)

    axes[1].set_title("Precision-Recall Curve", fontsize=18, weight="bold")
    axes[1].set_xlabel("Recall", fontsize=12, weight="bold")
    axes[1].set_ylabel("Precision", fontsize=12, weight="bold")

    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    axes[1].legend()

    # 3. ROC Curve ----------------------------

    axes[2].plot(fpr, tpr, linewidth=2.5, label=f"ROC AUC = {roc_auc:.4f}")

    # Random classifier line
    axes[2].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Random Classifier",
    )

    axes[2].fill_between(fpr, tpr, alpha=0.2)

    axes[2].set_title("ROC Curve", fontsize=18, weight="bold")
    axes[2].set_xlabel("False Positive Rate", fontsize=12, weight="bold")
    axes[2].set_ylabel("True Positive Rate", fontsize=12, weight="bold")

    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)

    axes[2].legend(loc="lower right")

    # Turn on grid for all plots
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

    # Plot to W&B if active run is provided
    if active_run is None:
        print("\n[INFO] No active W&B run provided. Skipping logging plots to W&B.\n")
        return

    print(f"\n[INFO] Logging plots to W&B run: {active_run.name}...\n")
