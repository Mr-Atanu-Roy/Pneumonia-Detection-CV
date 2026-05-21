import random
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import auc

from .transform import IMAGENET_MEAN, IMAGENET_STD


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize a tensor of shape (C, H, W) and converts to numpy array of shape (H, W, C). Returns a numpy array suitable for plotting.

    Normalization: (input - mean) / std
    Denormalization: input = (output × std) + mean

    Args:
        - tensor (torch.Tensor): tensor to be denormalized

    Returns:
        - np.ndarray: denormalized tensor

    """

    # make the STD & MEAN tensor obj
    std = torch.tensor(IMAGENET_STD)
    mean = torch.tensor(IMAGENET_MEAN)

    # clone to avoid modifying original tensor
    image = tensor.clone()

    # denormalize: (image * std) + mean
    image = image * std.view(3, 1, 1) + mean.view(3, 1, 1)

    image = image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    return image


def plot_data(
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    k: int = 10,
    title: str = "Random Sample From Images",
) -> None:
    """
    Plots k samples for the given dataloader

    Args:
        - dataloader (torch.utils.data.DataLoader): dataloader to plot
        - k (int): number of samples to plot

    Raise:
        - ValueError: if k is not a multiple of 2 or greater than 10
    """

    # k must be multiple of 2 and should be <= 10
    if k % 2 != 0 or k > 10:
        raise ValueError("k must be a multiple of 2 and should be <= 10")

    img, labels = next(iter(dataloader))

    # get k random samples from the datasetloader
    random_idx = random.sample(range(len(img)), k)

    # plot the samples
    fig, axes = plt.subplots(nrows=2, ncols=k // 2, figsize=(12, 7))
    axes = axes.flatten()  # flatten to 1D array

    for i, idx in enumerate(random_idx):
        axes[i].imshow(denormalize(img[idx]))  # denormalize before plotting

        axes[i].set_title(f"{class_names[labels[idx]]}")
        axes[i].axis("off")

    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.show()


def plot_and_log_evaluation_result(
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

    # !! TODO: log to wandb


def plot_grad_cams(
    model: torch.nn.Module,
    target_layers: List[torch.nn.Module],
    samples: Dict[str, Dict[str, torch.Tensor]],
    class_names: List[str],
    device: Optional[str] = None,
    only_grad_cam: bool = False,
):
    """
    Plots Grad-CAM, GradCAM++, and EigenCAM for K TP, FP, FN, TN cases random samples from the dataloader. If only_grad_cam is True, only plots Grad-CAM.

    Args:
        - model (torch.nn.Module): The trained model for which Grad-CAM is to be computed.
        - target_layers (List[torch.nn.Module]): List of layers for which to compute Grad-CAM.
        - samples (Dict[str, Dict[str, torch.Tensor]]): Dictionary of sample dictionaries containing "transformed_image_tensor", "true_label", and "pred_prob" for each of the four cases (TP, FP, FN, TN).
         Each case (TP, FP, FN, TN) should have a dictionary with the following keys:
            - "transformed_image_tensor": Tensor of shape (K, C, H, W) containing the transformed images ready for model input.
            - "true_label": Tensor of shape (K,) containing the true labels for the samples.
            - "pred_prob": Tensor of shape (K,) containing the predicted probabilities for the positive class for the samples.
        - class_names (List[str]): List of class names corresponding to the labels.
        - k (int): The number of random samples to plot for each case (TP, FP, FN, TN). Defaults to 5.
        - device (str, optional): The device to run the computations on (e.g., "cuda" or "cpu"). If None, automatically selects "cuda" if available. Defaults to None.
        - only_grad_cam (bool): If True, only plots Grad-CAM. If False, plots Grad-CAM, GradCAM++, and EigenCAM. Defaults to False.
    """

    # Set device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Initialize Grad-CAM with the model and target layers
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
    )

    # Copy the sample dictionary to avoid modifying the original
    samples_copy = samples.copy()

    # calculate Grad-CAM for each case and store it in samples_copy[case]["grad_cam"]
    for case, sample_dict in samples_copy.items():
        samples_copy[case]["grad_cam"] = _calculate_grad_cam(
            cam, sample_dict["transformed_image_tensor"]
        )

    # Plot the original and Grad-CAM images for each case over a grid of 4 rows (TP, FP, FN, TN) and K columns (random samples). Where each item in row will have: original image and Grad-CAM image side by side.
    plot_gradcam_samples(
        samples=samples_copy,
        class_names=class_names,
    )


def _calculate_grad_cam(cam: GradCAM, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Helper function to calculate Grad-CAM for a given input tensor.
    """

    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension if missing

    return cam(input_tensor=input_tensor, targets=None)


def plot_gradcam_samples(
    samples: Dict[str, Dict[str, torch.Tensor]],
    class_names: List[str],
    figsize_scale: Tuple[int, int] = (5, 4),
):
    """
    Plot GradCAM visualizations for TP, FP, FN, TN samples.

    Expected structure:
    samples_copy = {
        "TP": {
            "transformed_image_tensor": Tensor[B, C, H, W],
            "true_label": Tensor[B],
            "pred_label": Tensor[B],
            "pred_prob": Tensor[B],
            "grad_cam": Tensor/ndarray[B, H, W]
        },
        ...
    }
    """

    print("\n[INFO] Plotting Grad-CAM visualizations for TP, FP, FN, TN samples...\n")

    sns.set_theme(style="white")

    cases = ["TP", "FP", "FN", "TN"]

    # Maximum samples among all cases
    num_samples = max(
        samples[case]["transformed_image_tensor"].shape[0] for case in cases
    )

    num_cases = len(cases)

    # 2 columns per sample -> Original + GradCAM
    total_cols = num_samples * 2

    fig, axes = plt.subplots(
        nrows=num_cases,
        ncols=total_cols,
        figsize=(total_cols * figsize_scale[0], num_cases * figsize_scale[1]),
        squeeze=False,
    )

    fig.suptitle(
        "GradCAM Visualization of Samples",
        fontsize=25,
        fontweight="bold",
        y=1.02,
    )

    for row_idx, case in enumerate(cases):
        data = samples[case]

        images = data["transformed_image_tensor"]
        true_labels = data["true_label"]
        pred_labels = data["pred_label"]
        pred_probs = data["pred_prob"]
        gradcams = data["grad_cam"]

        batch_size = images.shape[0]

        # Row heading
        axes[row_idx, 0].text(
            -0.35,
            0.5,
            case,
            fontsize=22,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
            transform=axes[row_idx, 0].transAxes,
        )

        for sample_idx in range(num_samples):
            orig_ax = axes[row_idx, sample_idx * 2]
            cam_ax = axes[row_idx, sample_idx * 2 + 1]

            # Empty slots
            if sample_idx >= batch_size:
                orig_ax.axis("off")
                cam_ax.axis("off")
                continue

            single_img = images[sample_idx]
            grayscale_cam = gradcams[sample_idx]

            if torch.is_tensor(grayscale_cam):
                grayscale_cam = grayscale_cam.detach().cpu().numpy()

            true_label = int(true_labels[sample_idx].item())
            pred_label = int(pred_labels[sample_idx].item())
            pred_prob = float(pred_probs[sample_idx].item())

            # Denormalized image
            denorm_img = denormalize(single_img)

            # GradCAM overlay
            cam_image = show_cam_on_image(
                denorm_img,
                grayscale_cam,
                use_rgb=True,
            )

            metadata_text = (
                f"Actual: {class_names[true_label]}\n"
                f"Predicted: {class_names[pred_label]}\n"
                f"Prediction Probability: {pred_prob:.3f}"
            )

            # ---------------- ORIGINAL ----------------
            orig_ax.imshow(denorm_img)

            orig_ax.set_title(
                "Original",
                fontsize=14,
                fontweight="bold",
            )

            orig_ax.text(
                0.5,
                -0.12,
                metadata_text,
                fontsize=11,
                ha="center",
                va="top",
                transform=orig_ax.transAxes,
            )

            orig_ax.axis("off")

            # ---------------- GRADCAM ----------------
            cam_ax.imshow(cam_image)

            cam_ax.set_title(
                "GradCAM",
                fontsize=14,
                fontweight="bold",
            )

            cam_ax.text(
                0.5,
                -0.12,
                metadata_text,
                fontsize=11,
                ha="center",
                va="top",
                transform=cam_ax.transAxes,
            )

            cam_ax.axis("off")

    plt.tight_layout()
    plt.show()
