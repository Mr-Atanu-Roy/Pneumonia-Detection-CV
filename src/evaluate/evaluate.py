import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import wandb
from PIL import Image
from tqdm.auto import tqdm

from ..models import models
from ..plots import plot_and_log_evaluation_result, plot_grad_cams
from ..trainer.experiment import create_loss_fn
from ..transform import test_transforms
from ..utils import download_wandb_artifact, load_config
from .evaluation_metrics import EvaluationMetrics


def evaluate_model_checkpoint(
    model_name: str,
    model_checkpoint_name: str,
    dataloader: torch.utils.data.DataLoader,
    threshold: float = 0.5,
    model_artifact_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    log_to_wandb: bool = True,
    run_name: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
    use_seed: bool = False,
) -> Dict[str, float]:
    """
    Evaluates the given model checkpoint on the provided dataloader and saves the results into wandb.
    Computes: CM, Loss, ROC, AUROC, PR Curve, F1 Score, Precision, Recall, Specificity, and a Composite metric (weighted average of AUROC and F1 Score).

    Args:
        - model_name (str): Name of the model (eg: "resNet50").
        - model_checkpoint_name (str): Name of the model checkpoint file (e.g. "resnet50-EP5-B32.pth"). Download the checkpoint from W&B if not found locally.
        - dataloader (torch.utils.data.DataLoader): DataLoader for multiple samples to evaluate on.
        - threshold (float, optional): Classification threshold for converting predicted probabilities to binary labels.    Default is 0.5.
        - model_artifact_dir (str or Path): Local directory where model checkpoints are stored. Defaults to the "models" subdirectory within the artifacts directory specified in the config file.
        - device (str, optional): Device to run the evaluation on (e.g. "cuda" or "cpu"). If None, automatically selects "cuda" if available.
        - log_to_wandb (bool, optional): Whether to log the evaluation results to W&B. Defaults to True.
        - run_name (str, optional): Name for the W&B run when logging evaluation results. Defaults to the model checkpoint name_current_datetime.
        - wandb_tags (List[str], optional): List of additional tags to add to the W&B run. Defaults to None.
        - use_seed (bool, optional): If True, uses a fixed seed (42) for reproducible sample selection for Grad-CAM visualization. Defaults to False.

    Returns:
        - Dict[str, float]: Evaluation metrics

    Raises:
        - FileNotFoundError: If the checkpoint file is not found locally and cannot be downloaded
        - ValueError: If the checkpoint file does not contain a 'model_state_dict' key
        - ValueError: If the model_name and checkpoint model name do not match
    """

    model_checkpoint_name = Path(
        model_checkpoint_name
    ).stem  # remove .pth suffix if given

    # set device and run name defaults
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    run_name = (
        run_name
        or f"eval_{model_checkpoint_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    print(f"\n{'-' * 80}")

    print(f"PHASE — Evaluating  |  {model_name}")
    print(f"Checkpoint: {model_checkpoint_name}")

    print(f"{'-' * 80}\n")

    print(f"[INFO] Using device {device}\n")

    # load config for default paths and W&B project name
    cfg = load_config()
    model_artifact_dir = Path(
        model_artifact_dir or Path(cfg["paths"]["artifacts_dir"]) / "models"
    )

    project_name = cfg["wandb"]["project_name"]

    class_names = cfg["data"]["class_names"]

    pos_weight = torch.tensor(cfg["data"]["pos_weight"], device=device)
    loss_fn = create_loss_fn(name="binary_ce", pos_weight=pos_weight)

    # download the model checkpoint from W&B if not found locally
    checkpoint_path = download_wandb_artifact(
        artifact_name=model_checkpoint_name,
        artifact_type="model",
        local_download_path=model_artifact_dir,
    )

    # Load the model checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    else:
        raise ValueError(
            f"Checkpoint '{model_checkpoint_name}' does not contain 'model_state_dict'."
        )

    # Validate that the model name in the checkpoint matches the provided model_name
    if "model_name" in checkpoint and checkpoint["model_name"] != model_name:
        raise ValueError(
            f"Model name in checkpoint ('{checkpoint['model_name']}') does not match the provided model_name ('{model_name}')."
        )

    model = models.create_model(model_name=model_name)
    model.load_state_dict(model_state_dict)

    try:
        if log_to_wandb:
            # setup extra imp. wandb tags
            extra_wandb_tags = [
                "evaluation",
                str(model_checkpoint_name),
                model_name,
                model_checkpoint_name,
            ]

            # W&B initializations
            wandb.init(
                project=project_name,
                name=run_name,
                tags=(wandb_tags or []) + extra_wandb_tags,
                config={
                    "model_name": model_name,
                    "device": device,
                    "threshold": threshold,
                    "checkpoint_name": model_checkpoint_name,
                },
            )

        # Perform evaluation and compute metrics
        eval_metrics = _evaluate_step(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            device=device,
            threshold=threshold,
        )
        print("[INFO] Model Results: ")
        print(f"Model Name: {model_name} | Checkpoint Name: {model_checkpoint_name}")
        print(
            f"Recall: {eval_metrics['recall']:.5f} | Precision: {eval_metrics['precision']:.5f} | F1 Score: {eval_metrics['f1_score']:.5f} | AUROC: {eval_metrics['auroc']:.5f} | Specificity: {eval_metrics['specificity']:.5f} | Composite Score: {eval_metrics['composite']:.5f}"
        )

        # CM, ROC, PR Curve plotting and logging to W&B
        plot_and_log_evaluation_result(
            cm=eval_metrics["confusion_matrix"]["cm"],
            pr_curve=eval_metrics["precision_recall_curve"],
            roc_curve=eval_metrics["roc_curve"],
            class_names=class_names,
            model_name=model_name,
            true_labels=eval_metrics["all_true_labels"],
            pred_probs=eval_metrics["all_pred_probs"],
            active_run=wandb.run if log_to_wandb else None,
        )

        # unfreeze last model layer for Grad-CAM visualization
        models.unfreeze_for_finetune(model_name=model_name, model=model, n_layers=1)

        # get K random samples for each of the 4 cases: TP, FP, FN, TN for Grad-CAM visualization
        selected_samples = _get_selected_samples(
            dataloader=dataloader,
            true_label=torch.tensor(eval_metrics["all_true_labels"]),
            pred_label=torch.tensor(eval_metrics["all_pred_labels"]),
            pred_probs=torch.tensor(eval_metrics["all_pred_probs"]),
            k=5,
            use_seed=use_seed,
        )

        # get 5 fixed samples from the dataset (model-independent, always seeded)
        # These are the same across all model evaluations for cross-model comparison.
        fixed_samples = _get_fixed_samples(
            dataloader=dataloader,
            pred_label=torch.tensor(eval_metrics["all_pred_labels"]),
            pred_probs=torch.tensor(eval_metrics["all_pred_probs"]),
            k=5,
        )

        plot_grad_cams(
            model=model,
            target_layers=_resolve_target_layers(model_name=model_name, model=model),
            samples=selected_samples,
            fixed_samples=fixed_samples,
            class_names=class_names,
            model_name=model_name,
            device=device,
            active_run=wandb.run if log_to_wandb else None,
        )

    finally:
        # cleanup: finish the W&B run
        if log_to_wandb and wandb.run is not None:
            try:
                wandb.finish()
            except Exception:
                pass

    return eval_metrics


def _evaluate_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: str,
    threshold: float,
) -> Dict[str, Any]:
    """
    Helper function to perform the evaluation step for a model checkpoint. Called by evaluate_model_checkpoint().
    """

    model.to(device)
    model.eval()

    # Initialize metrics
    metrics = EvaluationMetrics().to(device)
    metrics.reset()

    pbar = tqdm(dataloader, desc="Evaluating")

    # Evaluate the model and compute metrics
    with torch.inference_mode():
        for image, label in pbar:
            image, label = image.to(device), label.to(device)

            label_float = label.float()
            label_long = label.long()

            logits = model(image).squeeze(1)
            loss = loss_fn(logits, label_float)

            pred_probs = torch.sigmoid(logits)
            pred_labels = (pred_probs >= threshold).long()

            metrics.update(
                loss=loss,
                pred_logits=logits,
                pred_labels=pred_labels,
                true_labels=label_long,
            )

    return metrics.compute()


def _resolve_target_layers(
    model_name: str, model: torch.nn.Module
) -> List[torch.nn.Module]:
    """
    Helper function to return the target layers for Grad-CAM based on the model name and architecture.
    """

    model_name = model_name.lower()

    target_layer_resolvers = {
        "resnet50": lambda m: [m.layer4[-1]],
        "densenet121": lambda m: [m.features.denseblock4],
        "efficientnet_b2": lambda m: [m.features[-1]],
        "vit_b_16": lambda m: [m.blocks[-1].norm1],
    }

    if model_name in target_layer_resolvers:
        return target_layer_resolvers[model_name](model)

    raise ValueError(
        f"Unsupported model_name '{model_name}' for Grad-CAM target layer resolution."
    )


def _get_selected_samples(
    dataloader: torch.utils.data.DataLoader,
    true_label: torch.Tensor,
    pred_label: torch.Tensor,
    pred_probs: torch.Tensor,
    k: int = 5,
    use_seed: bool = False,
):
    """
    Helper function to get K random samples for each of the 4 cases: TP, FP, FN, TN called by evaluate_model_checkpoint().
    Returns: Dict with keys "TP", "FP", "FN", "TN" and values as list of tuples (image_path, label, pred_prob) for K random samples of each case.
    """

    # Use a local RNG instance to avoid polluting global random state.
    # Seed is fixed at 42 when use_seed=True for full reproducibility.
    rng = random.Random(42 if use_seed else None)

    # Get the list of index for all cases
    tp_idx = torch.where((true_label == 1) & (pred_label == 1))[0]
    fn_idx = torch.where((true_label == 1) & (pred_label == 0))[0]
    fp_idx = torch.where((true_label == 0) & (pred_label == 1))[0]
    tn_idx = torch.where((true_label == 0) & (pred_label == 0))[0]

    # Randomly select K indices from above list
    # If list is of length > K then select K samples otherwise select whole list
    random_tp_idx = (
        rng.sample(tp_idx.tolist(), k) if len(tp_idx) > k else tp_idx.tolist()
    )
    random_fn_idx = (
        rng.sample(fn_idx.tolist(), k) if len(fn_idx) > k else fn_idx.tolist()
    )
    random_fp_idx = (
        rng.sample(fp_idx.tolist(), k) if len(fp_idx) > k else fp_idx.tolist()
    )
    random_tn_idx = (
        rng.sample(tn_idx.tolist(), k) if len(tn_idx) > k else tn_idx.tolist()
    )

    data_samples = dataloader.dataset.samples

    return {
        "TP": _build_batch(random_tp_idx, data_samples, pred_probs, pred_label),
        "FP": _build_batch(random_fp_idx, data_samples, pred_probs, pred_label),
        "FN": _build_batch(random_fn_idx, data_samples, pred_probs, pred_label),
        "TN": _build_batch(random_tn_idx, data_samples, pred_probs, pred_label),
    }


def _build_batch(
    indices: List[int],
    data_samples: List[Tuple[str, int]],
    pred_probs: torch.Tensor,
    pred_label: torch.Tensor,
):

    return {
        # "image_paths": [data_samples[i][0] for i in indices],
        "transformed_image_tensor": torch.stack(
            [
                test_transforms(
                    image=np.array(Image.open(data_samples[i][0]).convert("RGB"))
                )["image"]
                for i in indices
            ]
        ),
        "true_label": torch.stack([torch.tensor(data_samples[i][1]) for i in indices]),
        "pred_prob": torch.stack([pred_probs[i] for i in indices]),
        "pred_label": torch.stack([pred_label[i] for i in indices]),
    }


def _get_fixed_samples(
    dataloader: torch.utils.data.DataLoader,
    pred_label: torch.Tensor,
    pred_probs: torch.Tensor,
    k: int = 5,
) -> Dict[str, torch.Tensor]:
    """
    Selects K fixed samples from the dataset using a hardcoded seed (42).
    The selection is independent of model predictions, so the same images
    are chosen across all model evaluations for cross-model Grad-CAM comparison.

    Args:
        - dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        - pred_label (torch.Tensor): Predicted binary labels for all samples.
        - pred_probs (torch.Tensor): Predicted probabilities for all samples.
        - k (int): Number of fixed samples to select. Defaults to 5.

    Returns:
        - Dict[str, torch.Tensor]: Dictionary with keys "transformed_image_tensor",
          "true_label", "pred_prob", "pred_label" for the K fixed samples.
    """

    data_samples = dataloader.dataset.samples
    total = len(data_samples)

    # Always use seed 42 so the same indices are picked regardless of model
    rng = random.Random(42)
    fixed_indices = rng.sample(range(total), min(k, total))

    return _build_batch(fixed_indices, data_samples, pred_probs, pred_label)
