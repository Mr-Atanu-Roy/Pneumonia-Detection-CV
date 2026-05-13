"""
Handles Fine-Tuning experiments, including Cases B (continuation), C (from checkpoint), and D (standalone).

"""

from typing import Any, Dict, List, Optional

import torch

from ..utils import build_ft_suffix, build_standalone_ft_run_name, build_wandb_config
from .loop import train


def run_fine_tuning_experiment(
    model_name: str,
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    ft_optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    ft_epochs: int,
    tf_lr: float,
    ft_lr: float,
    n_layers: int,
    batch_size: int,
    tl_run_name: Optional[str],
    tl_checkpoint_path: Optional[str],
    from_checkpoint: bool,
    extra_config: Optional[Dict[str, Any]],
    artifacts_dir: str,
    project_name: str,
    device: str,
    best_model_metric_name: str,
    recall_threshold: float,
    extra_wandb_tags: Optional[List] = None,
) -> Dict:
    """
    Handles Fine-Tuning — Cases B (continuation), C (from checkpoint), D (standalone).

    - Always receives a fully prepared model — instantiated, weighted,
    and unfrozen.
    - Never loads checkpoints, unfreezes layers, or creates optimizers.

    Run name logic:
        Case B/C (tl_run_name given):
            tl_run_name + __FT_LR=...-EP=...-N_LY=...
            Case C additionally appends _CP to denote checkpoint origin.
        Case D (tl_run_name is None):
            resnet50-FT_LR=1e-5-EP=5-N_LY=2

    W&B Config traceability:
        Layer 1 (config) : tl_run_name stored in ft_config
        Layer 2 (tag)    : "TL_FT" tag when linked to a TL run
        Layer 3 (config) : tl_checkpoint_path stored in ft_config

    Args:
        - from_checkpoint (bool): True for Case C (loaded from saved .pth),
                                  False for Case B (continuation) and Case D.
                                  Controls the _CP suffix and tags.
    """

    # FT run name
    if tl_run_name is not None:
        # Cases B and C — FT name references TL run name
        ft_run_name = tl_run_name + build_ft_suffix(ft_lr, ft_epochs, n_layers)
        if from_checkpoint:
            ft_run_name += "_CP"  # suffix denotes this run loaded from a saved file
    else:
        # Case D — standalone FT, no TL parent
        ft_run_name = build_standalone_ft_run_name(
            model_name=model_name, ft_lr=ft_lr, ft_epochs=ft_epochs, n_layers=n_layers
        )

    # WandB config
    ft_config = build_wandb_config(
        model_name=model_name,
        mode="fine_tuning",
        epochs=ft_epochs,
        batch_size=batch_size,
        tf_lr=tf_lr,
        ft_lr=ft_lr,
        n_layers=n_layers,
        ft_epochs=ft_epochs,
        load_checkpoint=tl_checkpoint_path,
        extra=extra_config,
    )
    ft_config["phase"] = "FT"
    ft_config["tl_run_name"] = tl_run_name or "none"  # Layer 1
    ft_config["tl_checkpoint_path"] = tl_checkpoint_path or "none"  # Layer 3

    # WandB tags
    ft_tags = [model_name, "FT"]
    if tl_run_name is not None:
        ft_tags.append("TL_FT")
    if from_checkpoint:
        ft_tags.append("finetune_checkpoint-tf")

    # add any user-provided extra tags (can be a list of tags or None)
    if extra_wandb_tags:
        ft_tags.extend(extra_wandb_tags)

    print(f"\n{'-' * 80}")

    print(f"PHASE — Fine-Tuning  |  {model_name}")
    print(f"Run  : {ft_run_name}")
    print(f"Epochs: {ft_epochs}")
    if tl_run_name:
        print(f"Parent TL run  : {tl_run_name}")
    if tl_checkpoint_path:
        print(f"From checkpoint: {tl_checkpoint_path}")

    print(f"{'-' * 80}\n")

    ft_results = train(
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        model=model,
        loss_fn=loss_fn,
        optimizer=ft_optimizer,
        model_name=model_name,
        run_name=ft_run_name,
        config=ft_config,
        artifacts_dir=artifacts_dir,
        project_name=project_name,
        epochs=ft_epochs,
        device=device,
        best_model_metric_name=best_model_metric_name,
        recall_threshold=recall_threshold,
        wandb_tags=ft_tags,
    )

    return ft_results
