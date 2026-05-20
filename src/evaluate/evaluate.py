from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import wandb
from tqdm.auto import tqdm

from ..models import models
from ..utils import download_wandb_artifact, load_config
from .evaluation_metrics import EvaluationMetrics
from .plots import plot_and_log_curves


def evaluate_model_checkpoint(
    model_name: str,
    model_checkpoint_name: str,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    threshold: float = 0.5,
    model_artifact_dir: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
    log_to_wandb: bool = True,
    run_name: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluates the given model checkpoint on the provided dataloader and saves the results into wandb.
    Computes: CM, Loss, ROC, AUROC, PR Curve, F1 Score, Precision, Recall, Specificity, and a Composite metric (weighted average of AUROC and F1 Score).

    Args:
        - model_name (str): Name of the model (eg: "resNet50").
        - model_checkpoint_name (str): Name of the model checkpoint file (e.g. "resnet50-EP5-B32.pth"). Download the checkpoint from W&B if not found locally.
        - dataloader (torch.utils.data.DataLoader): DataLoader for multiple samples to evaluate on.
        - loss_fn (torch.nn.Module): Loss function to compute the loss on the evaluation dataset.
        - threshold (float, optional): Classification threshold for converting predicted probabilities to binary labels.    Default is 0.5.
        - model_artifact_dir (str or Path): Local directory where model checkpoints are stored. Defaults to the "models" subdirectory within the artifacts directory specified in the config file.
        - device (str, optional): Device to run the evaluation on (e.g. "cuda" or "cpu"). If None, automatically selects "cuda" if available.
        - log_to_wandb (bool, optional): Whether to log the evaluation results to W&B. Defaults to True.
        - run_name (str, optional): Name for the W&B run when logging evaluation results. Defaults to the model checkpoint name_current_datetime.

    Returns:
        - Dict[str, float]: Evaluation metrics

    Raises:
        - FileNotFoundError: If the checkpoint file is not found locally and cannot be downloaded
        - ValueError: If the checkpoint file does not contain a 'model_state_dict' key
        - ValueError: If the model_name and checkpoint model name do not match
    """

    # set device and run name defaults
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    run_name = (
        run_name
        or f"eval_{model_checkpoint_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # load config for default paths and W&B project name
    cfg = load_config()
    model_artifact_dir = Path(
        model_artifact_dir or Path(cfg["paths"]["artifacts_dir"]) / "models"
    )

    project_name = cfg["wandb"]["project_name"]

    # download the model checkpoint from W&B if not found locally
    checkpoint_path = download_wandb_artifact(
        artifact_name=Path(model_checkpoint_name).stem,
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

        # Plot and log metrics to W&B
        plot_and_log_curves(
            cm=eval_metrics["confusion_matrix"]["cm"],
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
