"""
Contains functions to train and eval model over a single epoch.
Also a print function to display results in beautiful formatted way.
"""

from typing import Dict

import torch
from tqdm.auto import tqdm

from .epoch_metrics import EpochMetrics


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: EpochMetrics,
    device: str,
) -> Dict:
    """
    Trains the given model over a single epoch

    Args:
        - model (torch.nn.Module): model to be trained
        - dataloader (torch.utils.data.DataLoader): dataloader for training data
        - loss_fn (torch.nn.Module): loss function for the model
        - optimizer (torch.optim.Optimizer): optimizer for the model
        - metrics (EpochMetrics): EpochMetrics object to update with training metrics
        - device (str): device to use for training

    Returns:
        - results (Dict): dictionary containing the training results for the epoch
    """

    metrics.reset()
    metrics.to(device)

    model.to(device)
    model.train()

    pbar = tqdm(dataloader, desc="Train", leave=False)  # tqdm Bar

    for image, label in pbar:
        image, label = image.to(device), label.to(device)

        label_float = label.float()
        label_long = label.long()

        logits = model(image).squeeze(1)
        loss = loss_fn(logits, label_float)

        preds = (logits >= 0).long()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics.update(
            loss=loss, pred_logits=logits, pred_labels=preds, true_labels=label_long
        )

        # adding live metrics to tqdm bar (computed before)
        pbar.set_postfix(
            {
                "f1": f"{metrics.f1_score.compute().item():.4f}",
                "loss": f"{metrics.loss.compute().item():.4f}",
            }
        )

    return metrics.compute()


def eval_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    metrics: EpochMetrics,
    device: str,
) -> Dict:
    """
    Evaluate the given model over a single epoch

    Args:
        - model (torch.nn.Module): model to be evaluated
        - dataloader (torch.utils.data.DataLoader): dataloader for evaluation data
        - loss_fn (torch.nn.Module): loss function for the model
        - metrics (EpochMetrics): EpochMetrics object to update with evaluation metrics
        - device (str): device to use for evaluation

    Returns:
        - results (Dict): dictionary containing the evaluation results for the epoch
    """

    metrics.reset()
    metrics.to(device)

    model.to(device)
    model.eval()

    pbar = tqdm(dataloader, desc="Eval ", leave=False)  # tqdm bar

    with torch.inference_mode():
        for image, label in pbar:
            image, label = image.to(device), label.to(device)

            label_float = label.float()
            label_long = label.long()

            logits = model(image).squeeze(1)
            loss = loss_fn(logits, label_float)

            preds = (logits >= 0).long()

            metrics.update(
                loss=loss, pred_logits=logits, pred_labels=preds, true_labels=label_long
            )

            # adding live metrics to tqdm bar (computed before)
            pbar.set_postfix(
                {
                    "f1": f"{metrics.f1_score.compute().item():.4f}",
                    "loss": f"{metrics.loss.compute().item():.4f}",
                }
            )

    return metrics.compute()
