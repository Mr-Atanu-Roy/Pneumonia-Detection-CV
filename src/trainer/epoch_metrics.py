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


@dataclass
class EpochMetrics:
    """
    Contains all TorchMetrics objects for a single epoch of training or evaluation.

    - loss: MeanMetrics - accumulates the mean loss over batches.
    - auroc: BinaryAUROC - updated with raw logits (raw model o/p), not preds
    - recall, precision, f1_score, specificity, accuracy: Updated with preds (after applying threshold)
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

    def compute(self, composite_weights: Optional[List[float]] = None) -> Dict:
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

        # set default composite weights if not provided
        composite_weights = composite_weights or [0.7, 0.3]

        # validate composite_weights len and sum
        if composite_weights is not None:
            if len(composite_weights) != 2:
                raise ValueError(
                    f"composite_weights must be a list of 2 floats. Given: {composite_weights}"
                )

            if not abs(sum(composite_weights) - 1.0) < 1e-6:
                raise ValueError(
                    f"composite_weights must sum to 1. Given: {composite_weights} with sum {sum(composite_weights)}"
                )

        # compute composite metrics separately than other metrics
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
