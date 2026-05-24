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
    model_name: str = "",
    figsize: Tuple[int, int] = (22, 6),
    true_labels: Optional[List[int]] = None,
    pred_probs: Optional[List[float]] = None,
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
        - model_name (str): Name of the model, included in plot titles. Defaults to "".
        - figsize (Tuple[int, int]): Figure size in inches.
                                     Defaults to (22, 6).
        - true_labels (List[int], optional): List of true labels for the samples. Required if active_run is provided for W&B logging. Defaults to None.
        - pred_probs (List[float], optional): List of predicted probabilities for the samples. Required if active_run is provided for W&B logging. Defaults to None.
        - active_run Optional[wandb.sdk.wandb_run.Run]: Optional active W&B run to log charts to. If None, charts will not be logged to W&B.
                                                        Defaults to None.

    Raises:
        - ValueError: If pr_curve or roc_curve dicts are missing required keys.
        - ValueError: If active_run is provided but true_labels or pred_probs are missing.
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

    # If active_run is provided, validate that true_labels and pred_probs are also provided for W&B logging
    if active_run is not None:
        if true_labels is None or pred_probs is None:
            raise ValueError(
                "If active_run is provided for W&B logging, true_labels and pred_probs must also be provided."
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

    axes[0].set_title(f"Confusion Matrix {model_name}", fontsize=18, weight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=12, weight="bold")
    axes[0].set_ylabel("True Label", fontsize=12, weight="bold")

    # Add TN, FP, FN, TP labels
    for text, label in zip(axes[0].texts, group_names):
        text.set_text(f"{text.get_text()}\n({label})")

    # 2. PR Curve ----------------------------

    axes[1].plot(recall, precision, linewidth=2.5, label=f"PR AUC = {pr_auc:.4f}")

    axes[1].fill_between(recall, precision, alpha=0.2)

    axes[1].set_title(
        f"Precision-Recall Curve {model_name}", fontsize=18, weight="bold"
    )
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

    axes[2].set_title(f"ROC Curve {model_name}", fontsize=18, weight="bold")
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

    log_plots_to_wandb(
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        evaluation_fig=fig,
        true_labels=true_labels,
        pred_probs=pred_probs,
        class_names=class_names,
        active_run=active_run,
    )

    print(f"\n[INFO] Logged plots to W&B run: {active_run.name}...\n")


def log_plots_to_wandb(
    pr_auc: float,
    roc_auc: float,
    evaluation_fig: plt.Figure,
    true_labels: List[int],
    pred_probs: List[float],
    class_names: List[str],
    active_run: wandb.run,
) -> None:
    """
    Plots the evaluation figure (confusion matrix, PR curve, ROC curve) to W&B and logs scalar metrics (PR AUC and ROC AUC) to the active W&B run for main visualization. Also logs PR Curve and ROC Curve using wandb.plot.pr_curve and wandb.plot.roc_curve for interactive visualization in W&B.

    Args:
        - pr_auc (float): Area Under the Precision-Recall Curve.
        - roc_auc (float): Area Under the ROC Curve.
        - evaluation_fig (plt.Figure): Matplotlib figure containing the confusion matrix, PR curve, and ROC curve.
        - true_labels (List[int]): List of true labels for the samples.
        - pred_probs (List[float]): List of predicted probabilities for the samples.
        - class_names (List[str]): List of class names corresponding to the labels.
        - active_run (wandb.run): Active W&B run to log charts to.
    """

    active_run.log(
        {
            # Entire evaluation panel
            "evaluation/evaluation_plots": wandb.Image(evaluation_fig),
            # Scalar metrics
            "evaluation/roc_auc": roc_auc,
            "evaluation/pr_auc": pr_auc,
        }
    )

    # wandb.plot.pr_curve / roc_curve expect y_probas as a 2D array of
    # shape (n_samples, n_classes). pred_probs is the probability for the
    # positive class (1D), so stack [1-p, p] to build the required shape.
    pred_probs_arr = np.array(pred_probs)
    y_probas_2d = np.column_stack([1 - pred_probs_arr, pred_probs_arr])

    # Log PR Curve
    active_run.log(
        {
            "evaluation/pr_curve": wandb.plot.pr_curve(
                y_true=true_labels,
                y_probas=y_probas_2d,
                labels=class_names,
            )
        }
    )

    # Log ROC Curve
    active_run.log(
        {
            "evaluation/roc_curve": wandb.plot.roc_curve(
                y_true=true_labels,
                y_probas=y_probas_2d,
                labels=class_names,
            )
        }
    )


def plot_grad_cams(
    model: torch.nn.Module,
    target_layers: List[torch.nn.Module],
    samples: Dict[str, Dict[str, torch.Tensor]],
    class_names: List[str],
    model_name: str = "",
    fixed_samples: Optional[Dict[str, torch.Tensor]] = None,
    device: Optional[str] = None,
    only_grad_cam: bool = False,
    active_run: Optional[wandb.sdk.wandb_run.Run] = None,
):
    """
    Plots Grad-CAM, GradCAM++, and EigenCAM for K TP, FP, FN, TN cases random samples from the dataloader. If only_grad_cam is True, only plots Grad-CAM.
    Optionally also plots Grad-CAM for fixed samples (model-independent) for cross-model comparison.

    Args:
        - model (torch.nn.Module): The trained model for which Grad-CAM is to be computed.
        - target_layers (List[torch.nn.Module]): List of layers for which to compute Grad-CAM.
        - samples (Dict[str, Dict[str, torch.Tensor]]): Dictionary of sample dictionaries containing "transformed_image_tensor", "true_label", and "pred_prob" for each of the four cases (TP, FP, FN, TN).
         Each case (TP, FP, FN, TN) should have a dictionary with the following keys:
            - "transformed_image_tensor": Tensor of shape (K, C, H, W) containing the transformed images ready for model input.
            - "true_label": Tensor of shape (K,) containing the true labels for the samples.
            - "pred_prob": Tensor of shape (K,) containing the predicted probabilities for the positive class for the samples.
        - class_names (List[str]): List of class names corresponding to the labels.
        - model_name (str): Name of the model, included in plot titles. Defaults to "".
        - fixed_samples (Optional[Dict[str, torch.Tensor]]): Dictionary containing fixed (model-independent) samples for cross-model Grad-CAM comparison.
          Expected keys: "transformed_image_tensor", "true_label", "pred_prob", "pred_label". If None, fixed sample plot is skipped. Defaults to None.
        - device (str, optional): The device to run the computations on (e.g., "cuda" or "cpu"). If None, automatically selects "cuda" if available. Defaults to None.
        - only_grad_cam (bool): If True, only plots Grad-CAM. If False, plots Grad-CAM, GradCAM++, and EigenCAM. Defaults to False.
    """

    # Set device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Initialize Grad-CAM with the model and target layers
    # For ViT, use vit_reshape_transform to reshape the feature maps for Grad-CAM
    if model_name.lower() == "vit_b_16":
        cam = GradCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=vit_reshape_transform,
        )
    else:
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
        model_name=model_name,
    )

    # Plot Grad-CAM for fixed (model-independent) samples for cross-model comparison
    if fixed_samples is not None:
        fixed_samples_copy = fixed_samples.copy()
        fixed_samples_copy["grad_cam"] = _calculate_grad_cam(
            cam, fixed_samples_copy["transformed_image_tensor"]
        )
        plot_gradcam_fixed_samples(
            samples=fixed_samples_copy,
            class_names=class_names,
            model_name=model_name,
            active_run=active_run,
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
    model_name: str = "",
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
        f"GradCAM Visualization of Samples {model_name}",
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


def plot_gradcam_fixed_samples(
    samples: Dict[str, torch.Tensor],
    class_names: List[str],
    model_name: str = "",
    figsize_scale: Tuple[int, int] = (5, 4),
    active_run: Optional[wandb.sdk.wandb_run.Run] = None,
):
    """
    Plot GradCAM visualizations for fixed (model-independent) samples. These samples are identical across all model evaluations, enabling direct cross-model Grad-CAM comparison.
    Also logs the figure to W&B if active_run is provided.

    Expected structure:
    samples = {
        "transformed_image_tensor": Tensor[K, C, H, W],
        "true_label": Tensor[K],
        "pred_label": Tensor[K],
        "pred_prob": Tensor[K],
        "grad_cam": Tensor/ndarray[K, H, W]
    }

    Args:
        - samples (Dict[str, torch.Tensor]): Dictionary with keys "transformed_image_tensor",
          "true_label", "pred_label", "pred_prob", and "grad_cam".
        - class_names (List[str]): List of class names corresponding to the labels.
        - figsize_scale (Tuple[int, int]): Scale factors for figure width and height per subplot. Defaults to (5, 4).
        - active_run (Optional[wandb.sdk.wandb_run.Run]): W&B run object for logging the figure. Defaults to None.
    """

    print(
        "\n[INFO] Plotting Grad-CAM visualizations for fixed cross-model comparison samples...\n"
    )

    sns.set_theme(style="white")

    images = samples["transformed_image_tensor"]
    true_labels = samples["true_label"]
    pred_labels = samples["pred_label"]
    pred_probs = samples["pred_prob"]
    gradcams = samples["grad_cam"]

    num_samples = images.shape[0]

    # 2 columns per sample -> Original + GradCAM
    total_cols = num_samples * 2

    fig, axes = plt.subplots(
        nrows=1,
        ncols=total_cols,
        figsize=(total_cols * figsize_scale[0], 1 * figsize_scale[1]),
        squeeze=False,
    )

    fig.suptitle(
        f"GradCAM — Fixed Samples (Cross-Model Comparison) {model_name}",
        fontsize=25,
        fontweight="bold",
        y=1.02,
    )

    for sample_idx in range(num_samples):
        orig_ax = axes[0, sample_idx * 2]
        cam_ax = axes[0, sample_idx * 2 + 1]

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

    # Log to W&B if active run is provided
    if active_run is not None:
        log_gradcam_fixed_samples_wandb(
            samples=samples,
            class_names=class_names,
            model_name=model_name,
            active_run=active_run,
        )


def log_gradcam_fixed_samples_wandb(
    samples: Dict[str, torch.Tensor],
    class_names: List[str],
    model_name: str = "",
    figsize: Tuple[int, int] = (8, 4),
    active_run: Optional[wandb.sdk.wandb_run.Run] = None,
):
    """
    Log the GradCAM visualizations for fixed cross-model comparison samples into W&B only.

    Each sample is logged separately:
        eval_gradcam/sample_1
        eval_gradcam/sample_2
        ...
    """

    sns.set_theme(style="white")

    images = samples["transformed_image_tensor"]
    true_labels = samples["true_label"]
    pred_labels = samples["pred_label"]
    pred_probs = samples["pred_prob"]
    gradcams = samples["grad_cam"]

    num_samples = images.shape[0]

    print(f"\n[INFO] Plotting and logging {num_samples} GradCAM samples...\n")

    for sample_idx in range(num_samples):
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        single_img = images[sample_idx]
        grayscale_cam = gradcams[sample_idx]

        if torch.is_tensor(grayscale_cam):
            grayscale_cam = grayscale_cam.detach().cpu().numpy()

        true_label = int(true_labels[sample_idx].item())
        pred_label = int(pred_labels[sample_idx].item())
        pred_prob = float(pred_probs[sample_idx].item())
        status = "Correct" if true_label == pred_label else "Incorrect"

        denorm_img = denormalize(single_img)

        cam_image = show_cam_on_image(
            denorm_img,
            grayscale_cam,
            use_rgb=True,
        )

        metadata_text = (
            f"Actual: {class_names[true_label]}\n"
            f"Predicted: {class_names[pred_label]}\n"
            f"Probability: {pred_prob:.3f}\n"
            f"Status: {status}"
        )

        axes[0].imshow(denorm_img)

        axes[0].set_title(
            "Original",
            fontsize=14,
            fontweight="bold",
        )

        axes[0].text(
            0.5,
            -0.15,
            metadata_text,
            fontsize=10,
            ha="center",
            va="top",
            transform=axes[0].transAxes,
        )

        axes[0].axis("off")

        axes[1].imshow(cam_image)

        axes[1].set_title(
            "GradCAM",
            fontsize=14,
            fontweight="bold",
        )

        axes[1].text(
            0.5,
            -0.15,
            metadata_text,
            fontsize=10,
            ha="center",
            va="top",
            transform=axes[1].transAxes,
        )

        axes[1].axis("off")

        fig.suptitle(
            f"GradCAM Sample {sample_idx + 1} — {model_name}",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()

        # Log to W&B if active run is provided
        if active_run is not None:
            active_run.log({f"eval_gradcam/sample_{sample_idx + 1}": wandb.Image(fig)})

        plt.close(fig)

    print("\n[INFO] GradCAM samples logged successfully.\n")


def vit_reshape_transform(
    tensor: torch.Tensor,
    height: int = 14,
    width: int = 14,
) -> torch.Tensor:
    """
    Reshape transform function for ViT models to be used with Grad-CAM.

    Args:
        - tensor (torch.Tensor): Input tensor of shape (batch_size, num_tokens, feature_dim).
        - height (int): Height of the feature map. Defaults to 14.
        - width (int): Width of the feature map. Defaults to 14.

    Input shape: [B, 197, C]
    Output shape: [B, C, 14, 14]

    The first token is the class token, and the remaining 196 tokens correspond to a 14x14 grid of patches (since 14*14=196). We discard the class token and reshape the remaining tokens into a 2D feature map.
    """

    # remove CLS token
    tensor = tensor[:, 1:, :]

    result = tensor.reshape(  # reshape to [B, 14, 14, C]
        tensor.shape[0],  # batch size
        height,
        width,
        tensor.shape[2],  # feature dimension
    )

    return result.permute(0, 3, 1, 2)  # [B, C, H, W]
