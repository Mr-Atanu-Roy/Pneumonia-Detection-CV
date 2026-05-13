"""
Contains the train function to train the model for given epochs
"""

from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional

import torch
import wandb
from tqdm.auto import tqdm

from ..utils import is_best_model, upload_artifacts_to_wandb
from .engine import eval_step, train_step


def train(
    train_dataloader: torch.utils.data.DataLoader,
    eval_dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    run_name: str,
    config: Dict[str, Any],
    artifacts_dir: str,
    project_name: str,
    epochs: int,
    device: str,
    best_model_metric_name: str,
    recall_threshold: float,
    wandb_tags: Optional[List[str]] = None,
) -> Dict:
    """
    Trains a model over given epochs with W&B experiment tracking and checkpointing.
    Note:
        - This training function is used for both future extraction and fine tuning by setting the model, optimizer.
        - Each call to train() is one independent W&B run (finetune after future extraction are 2 different runs)

    Args:
        - train_dataloader   : dataloader for training data
        - eval_dataloader    : dataloader for evaluation/validation data
        - model              : model to be trained
        - loss_fn            : loss function
        - optimizer          : optimizer
        - model_name         : architecture name, e.g. "resnet50" (used in       checkpoint filename & W&B)
        - run_name           : unique run identifier, e.g. "resnet50-tl"
        - config             : dict of hyperparameters to log in W&B (lr, batch_size, epochs, mode, etc.)
        - artifacts_dir      : directory where the best checkpoint .pth will be saved.
        - project_name       : W&B project name (default: "pneumonia-detection")
        - epochs             : number of training epochs
        - device             : device to train on
        - best_model_metric_name  : metric used to determine best model checkpoint (e.g. "auroc")
        - recall_threshold   : minimum recall threshold for saving model checkpoint
        - wandb_tags         : optional list of W&B tags to add to the run for better organization and filtering

    Returns:
        - results (Dict): train and eval metric dicts per epoch, plus path to best checkpoint
    """

    # W&B initializations
    wandb.init(
        project=project_name,
        name=run_name,
        tags=wandb_tags or [],
        config={"model_name": model_name, "epochs": epochs, "device": device, **config},
    )

    # create proper checkpoint path
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifacts_dir / f"{run_name}.pth"

    # store the train and eval metric of each epoch
    results = {"train": [], "eval": []}

    best_model_metric_value = 0.0  # checkpoint creation tracker for best model

    # tqdm bar
    pbar = tqdm(range(1, epochs + 1))

    start_time = timer()

    for epoch in pbar:
        pbar.set_description(f"Epoch [{epoch}/{epochs}]")

        train_results = train_step(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            dataloader=train_dataloader,
            device=device,
        )
        eval_results = eval_step(
            model=model, loss_fn=loss_fn, dataloader=eval_dataloader, device=device
        )

        # display the results
        _print_epoch_results(epoch, epochs, train_results, eval_results)

        # store the results
        results["train"].append(train_results)
        results["eval"].append(eval_results)

        # log metrics group by train and eval
        wandb.log(
            {
                "epoch": epoch,
                **{f"train/{k}": v for k, v in train_results.items()},
                **{f"eval/{k}": v for k, v in eval_results.items()},
            }
        )

        # model checkpoint. Save model if beats current best model on the best_model_metric_name and has recall above recall_threshold
        current_model_metric_value = (
            (eval_results["f1_score"], eval_results["auroc"])
            if best_model_metric_name == "composite"
            else eval_results[best_model_metric_name]
        )
        model_checkpoints = is_best_model(
            best_metric_name=best_model_metric_name,
            current_metric_value=current_model_metric_value,
            best_metric_value=best_model_metric_value,
            recall=eval_results["recall"],
            recall_threshold=recall_threshold,
        )

        best_model_metric_value = model_checkpoints["updated_best_metric_value"]
        if model_checkpoints["is_best"]:
            # save the model
            torch.save(
                obj={
                    "epoch": epoch,
                    "model_name": model_name,
                    "run_name": run_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_model_metric_name": best_model_metric_name,
                    "best_model_metric_value": best_model_metric_value,
                },
                f=checkpoint_path,
            )

            print(
                f"[INFO] New checkpoint with eval {best_model_metric_name} score={best_model_metric_value:.4f} & Recall={eval_results['recall']:.4f} saved at {checkpoint_path}"
            )
        print()

    end_time = timer()
    total_time = end_time - start_time
    print(
        f"Total training time: {total_time:.3f} seconds (~ {round(total_time / 60, 2)} minutes)\n"
    )

    # save model as W&B artifacts
    upload_artifacts_to_wandb(
        file_paths=[str(checkpoint_path)],
        names=[run_name],
        artifact_type="model",
        wandb_run_name=run_name,
        wandb_project=project_name,
        artifact_description=f"Best model checkpoint for {run_name} with eval {best_model_metric_name}={best_model_metric_value:.4f} & Recall={eval_results['recall']:.4f}",
        active_run=wandb.run,
    )

    wandb.finish()

    # store checkpoint path, best f1, total training time to results
    results["checkpoint_path"] = str(checkpoint_path)
    results["best_model_metric_name"] = best_model_metric_name
    results["best_model_metric_value"] = best_model_metric_value
    results["total_time_sec"] = total_time

    return results


def _print_epoch_results(
    epoch: int, epochs: int, train_results: Dict, eval_results: Dict
) -> None:
    """
    Display the train and test results of a single epoch in terminal in a formatted way
    """

    print(f"Epoch [{epoch}/{epochs}]")
    print("Train ", " | ".join(f"{k}: {v:.4f}" for k, v in train_results.items()))
    print("Eval  ", " | ".join(f"{k}: {v:.4f}" for k, v in eval_results.items()))
