from dataclasses import dataclass, fields
from typing import Any, Dict, List

import torch
from torchmetrics import BinaryConfusionMatrix, BinaryPrecisionRecallCurve, BinaryROC

from ..trainer.engine import EpochMetrics


@dataclass
class EvaluationMetrics:
    """
    Contains all evaluation metrics for a model checkpoint evaluation.
    - confusion_matrix: BinaryConfusionMatrix - updated with preds labels (after applying threshold)
    - precision_recall_curve: BinaryPrecisionRecallCurve - updated with raw logits (raw model o/p), not preds
    - roc_curve: BinaryROC - updated with raw logits (raw model o/p), not preds
    - epoch_metrics: Contains all the metrics from engine.py (loss, auroc, f1_score, recall, precision, specificity, accuracy)
    - all_probs: List of per-batch predicted probabilities (after sigmoid), concatenated at compute time
    - all_preds: List of per-batch predicted labels (after thresholding), concatenated at compute time
    """

    confusion_matrix: BinaryConfusionMatrix = None
    precision_recall_curve: BinaryPrecisionRecallCurve = None
    roc_curve: BinaryROC = None
    epoch_metrics: EpochMetrics = None
    all_probs: List[torch.Tensor] = None
    all_preds: List[torch.Tensor] = None

    # Fields that are plain Python lists, not torchmetrics objects
    _LIST_FIELDS = frozenset(
        {"all_probs", "all_preds"}
    )  # frozenset: immutable version of set

    def __post_init__(self):
        # initialize all metrics
        self.confusion_matrix = BinaryConfusionMatrix()
        self.precision_recall_curve = BinaryPrecisionRecallCurve()
        self.roc_curve = BinaryROC(thresholds=None)
        self.epoch_metrics = EpochMetrics()
        self.all_probs: List[torch.Tensor] = []
        self.all_preds: List[torch.Tensor] = []

    def to(self, device: str) -> "EvaluationMetrics":
        # move all torchmetrics to the specified device (list fields are skipped)

        for field in fields(self):
            if field.name in self._LIST_FIELDS:
                continue
            setattr(self, field.name, getattr(self, field.name).to(device))
        return self

    def reset(self) -> None:
        # reset all metrics to their initial state

        for field in fields(self):
            if field.name in self._LIST_FIELDS:
                setattr(self, field.name, [])
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
            metric = getattr(self, field.name)

            if field.name == "confusion_matrix":
                metric.update(pred_labels, true_labels)
            elif field.name == "epoch_metrics":
                metric.update(
                    loss=loss,
                    pred_logits=pred_logits,
                    pred_labels=pred_labels,
                    true_labels=true_labels,
                )
            elif field.name == "all_probs":
                self.all_probs.append(torch.sigmoid(pred_logits).detach().cpu())
            elif field.name == "all_preds":
                self.all_preds.append(pred_labels.detach().cpu())
            else:
                metric.update(pred_logits, true_labels)

    def compute(self) -> Dict[str, Any]:
        """
        Compute final metric values and return as a dictionary.

        Returns:
            - Dict[str, Any]: Dictionary containing all computed metrics
            Eg:
                {
                "composite": 0.55,
                "loss": 0.1234,
                "recall": 0.5678,
                "precision": 0.9101,
                "auroc": 0.1121,
                "f1_score": 0.3141,
                "specificity": 0.5161,
                "accuracy": 0.7181,
                "confusion_matrix": {
                    "cm": [[TN, FP], [FN, TP]],
                    "tn": 100,
                    "fp": 10,
                    "fn": 5,
                    "tp": 85
                },
                "precision_recall_curve": {
                    "precision": [...],
                    "recall": [...],
                    "thresholds": [...]
                },
                "roc_curve": {
                    "fpr": [...],
                    "tpr": [...],
                    "thresholds": [...]
                },
                "all_probs": Tensor(...),
                "all_preds": Tensor(...),
            }
        """

        # Compute epoch metrics
        epoch_metrics_dict = self.epoch_metrics.compute()

        # Compute confusion matrix
        cm_tensor = self.confusion_matrix.compute()
        tn, fp, fn, tp = cm_tensor.flatten()

        # Compute precision-recall curve
        precision, recall, thresholds = self.precision_recall_curve.compute()

        # Compute ROC curve
        fpr, tpr, roc_thresholds = self.roc_curve.compute()

        # Concatenate accumulated per-batch lists into single tensors
        all_probs = torch.cat(self.all_probs) if self.all_probs else torch.tensor([])
        all_preds = torch.cat(self.all_preds) if self.all_preds else torch.tensor([])

        return {
            **epoch_metrics_dict,
            "confusion_matrix": {
                "cm": cm_tensor.cpu().long().tolist(),
                "tn": tn.item(),
                "fp": fp.item(),
                "fn": fn.item(),
                "tp": tp.item(),
            },
            "precision_recall_curve": {
                "precision": precision.cpu().tolist(),
                "recall": recall.cpu().tolist(),
                "thresholds": thresholds.cpu().tolist()
                if thresholds is not None
                else None,  # threshold can be None
            },
            "roc_curve": {
                "fpr": fpr.cpu().tolist(),
                "tpr": tpr.cpu().tolist(),
                "thresholds": roc_thresholds.cpu().tolist()
                if roc_thresholds is not None
                else None,
            },
            "all_probs": all_probs,
            "all_preds": all_preds,
        }
