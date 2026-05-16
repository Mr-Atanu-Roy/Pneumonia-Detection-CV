"""
Contains functions to train and eval model over a single epoch.
Also a print function to display results in beautiful formatted way.
"""

from dataclasses import dataclass, fields
from typing import Dict, List, Optional

import torch
from torchmetrics import MeanMetric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)
from tqdm.auto import tqdm


@dataclass
class EpochMetrics:
    """
    Contains all TorchMetrics objects for a single epoch of training or evaluation.

    - loss: MeanMetrics - accumulates the mean loss over batches.
    - auroc: BinaryAUROC - updated with raw logits (raw model o/p), not preds
    - recall, precision, auroc, f1_score, specificity, accuracy: Updated with preds (after applying threshold)
    """

    composite: float = None
    loss: MeanMetric = None
    auroc: BinaryAUROC = None
    f1_score: BinaryF1Score = None
    recall: BinaryRecall = None
    precision: BinaryPrecision = None
    specificity: BinarySpecificity = None
    accuracy: BinaryAccuracy = None

    def __post_init__(self):
        # initialize all metrics

        self.loss = MeanMetric()
        self.auroc = BinaryAUROC()
        self.f1_score = BinaryF1Score(zero_division=0)
        self.recall = BinaryRecall(zero_division=0)
        self.precision = BinaryPrecision(zero_division=0)
        self.specificity = BinarySpecificity(zero_division=0)
        self.accuracy = BinaryAccuracy(zero_division=0)

    def to(self, device: str) -> "EpochMetrics":
        # move all metrics to the specified device

        for field in fields(self):
            
            if field.name == "composite":
                # for composite metric .to() method can't be called so skip it
                continue

            setattr(self, field.name, getattr(self, field.name).to(device))
        return self

    def reset(self) -> None:
        # reset all metrics to their initial state

        for field in fields(self):

            if field.name == "composite":
                # for composite metric .reset() method can't be called so skip it
                continue

            getattr(self, field.name).reset()

    def update(
        self,
        loss: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_labels: torch.Tensor,
        true_labels: torch.Tensor,
    ) -> None:
        """
        Update all metrics for a single batch.

        Args:
            - loss (torch.Tensor): loss for the batch
            - pred_logits (torch.Tensor): raw logits from the model (model's raw output)
            - pred_labels (torch.Tensor): predicted labels (after applying threshold)
            - true_labels (torch.Tensor): true labels
        """

        for field in fields(self):

            if field.name == "composite":
                # for composite metric .update() method can't be called so skip it
                continue

            metric = getattr(self, field.name)

            if field.name == "loss":
                metric.update(loss)

            elif field.name == "auroc":
                metric.update(pred_logits, true_labels)

            else:
                metric.update(pred_labels, true_labels)

    def compute(self, composite_weights: Optional[List[float]] = [0.7, 0.3]) -> Dict:
        """
        Compute the final metric values and return as a dictionary

        Args:
            - composite_weights (List[float]): weights for f1_score and auroc when calculating composite metric (Eg: 0.7 for f1 and 0.3 for auroc)

        Returns:
            - result (Dict): dictionary of computed metric values for the epoch
            Eg: {
                "composite": 0.55,
                "loss": 0.1234,
                "recall": 0.5678,
                "precision": 0.9101,
                "auroc": 0.1121,
                "f1_score": 0.3141,
                "specificity": 0.5161,
                "accuracy": 0.7181,
            }

        - NOTE: the composite is a derived metric so its only computed at the time of compute.
        """

        # compute composite metrics seperatly than other metrics
        result = {
            field.name: getattr(self, field.name).compute().item()
            for field in fields(self)
            if field.name != "composite"
        }

        result["composite"] = (
            composite_weights[0] * result["f1_score"]
            + composite_weights[1] * result["auroc"]
        )

        return result


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
