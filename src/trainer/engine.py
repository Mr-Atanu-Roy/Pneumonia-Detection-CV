"""
Contains functions to train and eval model over a single epoch.
Also a print function to display results in beautiful formatted way.
"""


import torch

from typing import Dict
from tqdm.auto import  tqdm

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    BinaryAUROC
)

def train_step(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device: str)->Dict:
    """
    Trains the given model over a single epoch

    Args:
        - model (torch.nn.Module): model to be trained
        - dataloader (torch.utils.data.DataLoader): dataloader for training data
        - loss_fn (torch.nn.Module): loss function for the model
        - optimizer (torch.optim.Optimizer): optimizer for the model
        - device (str): device to use for training

    Returns:
        - results (Dict): dictionary containing the training results for the epoch
    """

    metrics = {
        "recall": BinaryRecall(zero_division=0).to(device),
        "precision": BinaryPrecision(zero_division=0).to(device),
        "auroc": BinaryAUROC().to(device),
        "f1_score": BinaryF1Score(zero_division=0).to(device),
        "specificity": BinarySpecificity(zero_division=0).to(device),
        "accuracy": BinaryAccuracy(zero_division=0).to(device)
    }

    total_loss = 0.0

    model.to(device)
    model.train()

    pbar = tqdm(dataloader, desc="Train", leave=False)  #tqdm Bar

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

        # metric updation
        total_loss += loss.item()
        for metric in metrics.values():
            metric.update(preds, label_long)

        # adding live metrics to tqdm bar (computed before)
        pbar.set_postfix({
            "auroc": f"{metrics['auroc'].compute().item():.4f}",
            "loss": f"{loss.item():.4f}"
        })

    # compute the metrics
    results = {k: v.compute().item() for k, v in metrics.items()}
    results["loss"] = total_loss / len(dataloader)

    # reset all metrics
    for metric in metrics.values():
        metric.reset()

    return results


def eval_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: str)->Dict:
    """
    Evaluate the given model over a single epoch

    Args:
        - model (torch.nn.Module): model to be evaluated
        - dataloader (torch.utils.data.DataLoader): dataloader for evaluation data
        - loss_fn (torch.nn.Module): loss function for the model
        - device (str): device to use for evaluation

    Returns:
        - results (Dict): dictionary containing the evaluation results for the epoch
    """

    metrics = {
        "recall": BinaryRecall(zero_division=0).to(device),
        "precision": BinaryPrecision(zero_division=0).to(device),
        "auroc": BinaryAUROC().to(device),
        "f1_score": BinaryF1Score(zero_division=0).to(device),
        "specificity": BinarySpecificity(zero_division=0).to(device),
        "accuracy": BinaryAccuracy(zero_division=0).to(device)
    }

    total_loss = 0.0

    model.to(device)

    model.eval()
    with torch.inference_mode():

        pbar = tqdm(dataloader, desc="Eval ", leave=False)  #tqdm bar

        for image, label in pbar:

            image, label = image.to(device), label.to(device)

            label_float = label.float()
            label_long = label.long()

            logits = model(image).squeeze(1)
            loss = loss_fn(logits, label_float)

            preds = (logits >= 0).long()

            # update eval metrics
            total_loss += loss.item()
            for metric in metrics.values():
                metric.update(preds, label_long)

            # adding live metrics to tqdm bar (computed before)
            pbar.set_postfix({
                "auroc": f"{metrics['auroc'].compute().item():.4f}",
                "loss": f"{loss.item():.4f}"
            })


    # compute the metrics
    results = {k: v.compute().item() for k, v in metrics.items()}
    results["loss"] = total_loss / len(dataloader)

    # reset the metrics
    for metric in metrics.values():
        metric.reset()

    return results


def print_epoch_results(epoch: int,
                        epochs: int,
                        train_results: Dict,
                        eval_results: Dict)->None:
    """
    Display the train and test results of a single epoch in terminal in a formated way
    """

    print(f"Epoch [{epoch}/{epochs}]")
    print("Train ", " | ".join(f"{k}: {v:.4f}" for k, v in train_results.items()))
    print("Eval  ", " | ".join(f"{k}: {v:.4f}" for k, v in eval_results.items()))
