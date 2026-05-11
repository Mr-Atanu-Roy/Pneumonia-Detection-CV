"""
Handles Transfer Learning experiments, including both Case A (feature learning only) and Case B (feature learning followed by fine-tuning).

"""

from typing import Any, Dict, List, Optional

import torch

from ..models.models import create_optimizer, unfreeze_for_finetune
from ..utils import build_ft_suffix, build_tl_run_name, build_wandb_config
from .fine_tuning_experiment import run_fine_tuning_experiment
from .loop import train


def run_transfer_learning_experiment(
    model_name: str,
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    tf_optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    batch_size: int,
    optimizer_name: str,
    tf_lr: float,
    ft_lr: Optional[float],
    lr_decay: Optional[float],
    n_layers: Optional[int],
    ft_epochs: Optional[int],
    extra_config: Optional[Dict[str, Any]],
    artifacts_dir: str,
    project_name: str,
    device: str,
    extra_wandb_tags: Optional[List] = None,
) -> Dict:
    """
    Handles Transfer Learning — Case A and Case B.

    Case A (ft_epochs=None):
        Trains for `epochs` and returns tl_results directly.

    Case B (ft_epochs=<int>):
        Trains for `epochs`, reloads best TL checkpoint, unfreezes
        `n_layers` backbone blocks, re-creates optimizer with
        discriminative LRs, then delegates to run_fine_tuning_experiment().
        Returns {"tl": tl_results, "ft": ft_results}.
    """

    # WandB run name
    tl_run_name = build_tl_run_name(
        model_name=model_name, epochs=epochs, tf_lr=tf_lr, batch_size=batch_size
    )

    # pre build FT run name to add in TF config if TF is followed by FT
    ft_run_name = None
    if ft_epochs is not None:
        ft_run_name = tl_run_name + build_ft_suffix(ft_lr, ft_epochs, n_layers)

    # wandb config
    tl_config = build_wandb_config(
        model_name=model_name,
        mode="transfer_learning",
        epochs=epochs,
        batch_size=batch_size,
        tf_lr=tf_lr,
        ft_lr=ft_lr if ft_epochs else None,
        n_layers=n_layers if ft_epochs else None,
        ft_epochs=ft_epochs,
        load_checkpoint=None,
        extra=extra_config,
    )

    tl_config["phase"] = "TL"
    tl_config["has_ft_continuation"] = ft_epochs is not None

    if ft_run_name is not None:
        tl_config["ft_run_name"] = ft_run_name

    # WandB tags
    tl_tags = [model_name, "TL"]
    if ft_epochs is not None:
        tl_tags.append("TL_FT")

    # add any user-provided extra tags (can be a list of tags or None)
    if extra_wandb_tags:
        tl_tags.extend(extra_wandb_tags)

    print(f"\n{'-' * 80}")

    print(f"PHASE — Transfer Learning  |  {model_name}")
    print(f"Run: {tl_run_name}")
    print(f"Epochs: {epochs}")

    print(f"{'-' * 80}\n")

    # TL training
    tl_results = train(
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        model=model,
        loss_fn=loss_fn,
        optimizer=tf_optimizer,
        model_name=model_name,
        run_name=tl_run_name,
        config=tl_config,
        artifacts_dir=artifacts_dir,
        project_name=project_name,
        epochs=epochs,
        device=device,
        wandb_tags=tl_tags,
    )

    # Case A: return tl_results (only feature learning)
    if ft_epochs is None:
        return tl_results

    # Case B: prepare model and delegate to run_fine_tuning_experiment()
    tl_checkpoint_path = tl_results["checkpoint_path"]

    # reload best TL weights
    checkpoint = torch.load(tl_checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"\n[INFO] Reloaded best TL weights from '{tl_checkpoint_path}' "
        f"(epoch {checkpoint['epoch']}, AU-ROC: {checkpoint['best_auc']:.4f})"
    )

    # unfreeze last n_layers backbone blocks
    unfreeze_for_finetune(model_name, model, n_layers)
    print(f"[INFO] Unfroze last {n_layers} backbone block(s) for fine-tuning.")

    # re-create optimizer for fine tuning
    ft_optimizer = create_optimizer(
        model_name=model_name,
        model=model,
        optimizer_name=optimizer_name,
        mode="fine_tuning",
        n_layers=n_layers,
        tf_lr=tf_lr,
        ft_lr=ft_lr,
        lr_decay=lr_decay,
    )

    ft_results = run_fine_tuning_experiment(
        model_name=model_name,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        ft_optimizer=ft_optimizer,
        loss_fn=loss_fn,
        ft_epochs=ft_epochs,
        tf_lr=tf_lr,
        ft_lr=ft_lr,
        n_layers=n_layers,
        batch_size=batch_size,
        tl_run_name=tl_run_name,  # for run name construction, layer 1 link
        tl_checkpoint_path=tl_checkpoint_path,  # layer 3 link
        from_checkpoint=False,  # Case B — continuation, not checkpoint load
        extra_config=extra_config,
        artifacts_dir=artifacts_dir,
        project_name=project_name,
        device=device,
        extra_wandb_tags=extra_wandb_tags,
    )

    return {"tl": tl_results, "ft": ft_results}
