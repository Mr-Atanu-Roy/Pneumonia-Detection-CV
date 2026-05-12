"""
Contains the main entry point for training the model

run_experiment()
│
├── VALIDATION
│   ├── ValueError  → load_checkpoint AND ft_epochs both given
│   ├── ValueError → dataloaders AND dataloader params both given
│   └── ValueError  → mode not in ["transfer_learning", "fine_tuning"]
│
├── DATALOADER RESOLUTION
│   └── created using: create_dataloaders()
│       └── pos_weight: set as hyperparameters
│
├── run_name AUTO-CONSTRUCTION
│
├── CASE A: mode="transfer_learning", ft_epochs=None, load_checkpoint=None
│   ├── create_model → create_optimizer(mode="transfer_learning")
│   ├── train(epochs)
│   │   └── W&B Run:
│   │       ├── run_name: resnet50-TL_LR1e-3-EP8-B32
│   │       └── tags: [model_name, "TL"]
│   └── return tl_results
│
├── CASE B: mode="transfer_learning", ft_epochs given, load_checkpoint=None
│   ├── Phase 1 (Transfer Learning):
│   │   ├── create_model → create_optimizer(mode="transfer_learning")
│   │   ├── train(epochs)
│   │   │   └── W&B Run 1:
│   │   │       ├── run_name: resnet50-TL_LR1e-3-EP8-B32
│   │   │       └── tags: [model_name, "TL", "TL_FT"]
│   │
│   ├── unfreeze_for_finetune(n_layers)
│   ├── create_optimizer(mode="fine_tuning", lr=ft_lr)
│   │
│   ├── Phase 2 (Fine-Tuning):
│   │   ├── train(ft_epochs)
│   │   │   └── W&B Run 2:
│   │   │       ├── run_name:
|   |   |            resnet50-TL_LR1e-3-EP8-B32__FT_LR1e-5-EP5-N_LY2
│   │   │       ├── tags: [model_name, "FT", "TL_FT"]
│   │   │       └── config:
│   │   │           ├── tl_run_name
│   │   │           └── tl_checkpoint_path
│   │
│   └── return {"tl": tl_results, "ft": ft_results}
│
├── CASE C: mode="fine_tuning", load_checkpoint given, ft_epochs given
│   ├── create_model
│   ├── load state_dict from checkpoint
│   ├── unfreeze_for_finetune(n_layers)
│   ├── create_optimizer(mode="fine_tuning", lr=ft_lr)
│   ├── train(ft_epochs)
│   │   └── W&B Run:
│   │       ├── run_name:
|   |            resnet50-TL_LR1e-3-EP8-B32__FT_LR1e-5-EP5-N_LY2_CP
│   │       ├── tags: [model_name, "FT", "TL_FT", "finetune_checkpoint-tf"]
│   │       └── config:
│   │           ├── tl_run_name (from checkpoint)
│   │           └── tl_checkpoint_path
│   └── return ft_results
│
└── CASE D: mode="fine_tuning", load_checkpoint=None, ft_epochs given
    ├── create_model
    ├── unfreeze_for_finetune(n_layers)
    ├── create_optimizer(mode="fine_tuning", lr=ft_lr)
    ├── train(ft_epochs)
    │   └── W&B Run:
    │       ├── run_name: resnet50-FT_LR1e-5-EP5-N_LY2
    │       └── tags: [model_name, "FT"]
    └── return ft_results
"""

import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import wandb

from ..dataloader import create_dataloaders
from ..models.models import create_model, create_optimizer, unfreeze_for_finetune
from ..transform import test_transforms, train_transforms
from ..utils import load_config, set_seeds
from .fine_tuning_experiment import run_fine_tuning_experiment
from .transfer_learning_experiment import run_transfer_learning_experiment

set_seeds()

# Load config.yaml for default values. CLI args (in train.py) will override these defaults.

_cfg = load_config()

_project_name = _cfg["wandb"]["project_name"]

_train_val_dir = Path(_cfg["paths"]["train_val_dir"])
_model_artifacts_dir = Path(_cfg["paths"]["artifacts_dir"]) / "models"

_val_size = _cfg["data"]["val_size"]
_pos_weight = _cfg["data"]["pos_weight"]

_batch_size = _cfg["dataloader"]["batch_size"]
_epochs = _cfg["training"]["epochs"]
_device = _cfg["training"]["device"] or (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # if default is None, resolve to "cuda" if available else "cpu"

_num_workers = _cfg["dataloader"]["num_workers"] or os.cpu_count()
_persistent_workers = _cfg["dataloader"]["persistent_workers"] or None

_tf_lr = _cfg["optimizer"]["tf_lr"]
_ft_lr = _cfg["optimizer"]["ft_lr"]
_lr_decay = _cfg["optimizer"]["lr_decay"]
_n_layers = _cfg["optimizer"]["n_layers"]
_optimizer_name = _cfg["optimizer"]["optimizer_name"]

_best_model_metric = _cfg["training"]["best_model_metric"]
_recall_threshold = _cfg["training"]["recall_threshold"]


def run_experiment(
    model_name: str,
    mode: str = "transfer_learning",
    # W&B
    project_name: str = _project_name,
    extra_config: Optional[Dict[str, Any]] = None,
    # dataloaders: gets auto created
    train_val_dir: str = _train_val_dir,
    train_transform: Callable[[Any], Any] = train_transforms,
    test_transform: Callable[[Any], Any] = test_transforms,
    batch_size: int = _batch_size,
    val_size: float = _val_size,
    num_workers: int = _num_workers,
    persistent_workers: Optional[bool] = _persistent_workers,
    # loss fn
    pos_weight: float = _pos_weight,
    # training
    epochs: int = _epochs,
    artifacts_dir: str = _model_artifacts_dir,
    device: str = _device,
    # optimizer/model architecture
    optimizer_name: str = _optimizer_name,
    tf_lr: float = _tf_lr,
    ft_lr: float = _ft_lr,
    lr_decay: float = _lr_decay,
    n_layers: int = _n_layers,
    # fine tune control
    ft_epochs: Optional[int] = None,
    checkpoint_name: Optional[str] = None,
    # best model checkpointing control
    best_model_metric: str = _best_model_metric,
    recall_threshold: float = _recall_threshold,
    # Optional W&B tags from user
    extra_wandb_tags: Optional[List] = None,
):
    """
    Top-level experiment entry point.
    1. Validate all input parameters
    2. Resolve dataloaders (direct or auto-created)
    3. Resolve pos_weight (provided or computed)
    4. Build the loss function
    5. Route to _run_transfer_learning() or _run_fine_tuning()

    ── Cases ───────────────────────────────────────────────────────────
    Case A │ mode="transfer_learning", ft_epochs=None
           │ Feature extraction only. Returns tl_results.
           │
    Case B │ mode="transfer_learning", ft_epochs=<int>
           │ Feature extraction → fine-tuning.
           │ Returns {"tl": tl_results, "ft": ft_results}.
           │
    Case C │ mode="fine_tuning", checkpoint_name=<name>, ft_epochs=<int>
           │ Fine-tuning from existing checkpoint.
           │ Returns ft_results.
           │
    Case D │ mode="fine_tuning", checkpoint_name=None, ft_epochs=<int>
           │ Fine-tuning only on a freshly created model.
           │ Returns ft_results.

    Args:
        - model_name: name of the model to be used. Eg: resent50 (must be present in model_registry)
        - mode: "transfer_learning" or "fine_tuning"

        - project_name: W&B project name
        - extra_config: extra config to be passed to W&B

        - epochs: number of epochs to train for future extraction
        - artifacts_dir: directory where the best checkpoint .pth are saved
        - device: device to train on

        - tf_lr: transfer learning learning rate
        - ft_lr: fine tuning learning rate
        - lr_decay: learning rate decay factor
        - n_layers: number of layers to unfreeze for fine tuning

        - ft_epochs: number of epochs to train for fine tuning
        - checkpoint_name: name of the checkpoint to load for fine tuning. Eg: 'vit_b_16-TL_LR1e-4-EP5-B32.pth'

        - best_model_metric: metric used to determine the best model checkpoint (e.g. "auroc", "accuracy", etc.)
        - recall_threshold: minimum recall threshold for saving model checkpoint to ensure we are not over fitting

    Raises:
        - ValueError : num_workers = 0 and persistent_workers = True (invalid combination)
        - ValueError : mode is not "transfer_learning" or "fine_tuning"
        - ValueError : mode="transfer_learning" and checkpoint_name is given
        - ValueError : mode="fine_tuning" and ft_epochs = None
        - ValueError : best_model_metric is not in ("recall", "precision", "auroc", "f1_score", "specificity", "accuracy")
        - ValueError : recall_threshold is not between 0 and 1
        - ValueError : checkpoint_name given and model_name does not match saved model
    """

    # VALIDATE PARAMS---------------------

    # 1. Invalid combination of num_workers and persistent_workers
    if num_workers == 0 and persistent_workers:
        raise ValueError(
            "Invalid combination: num_workers=0 and persistent_workers=True. persistent_workers can only be True if num_workers > 0."
        )

    # 2. Invalid mode
    if mode not in ("transfer_learning", "fine_tuning"):
        raise ValueError(
            f"`mode` must be 'transfer_learning' or 'fine_tuning' ('{mode}' given)."
        )

    # 3. checkpoint_name is meaningless for transfer learning
    if mode == "transfer_learning" and checkpoint_name is not None:
        raise ValueError(
            "`checkpoint_name` cannot be used with mode='transfer_learning'. "
            "To fine-tune an existing checkpoint, use mode='fine_tuning'."
        )

    # 4. fine_tuning requires at least one of checkpoint_name or ft_epochs
    if mode == "fine_tuning" and ft_epochs is None:
        raise ValueError(
            "mode='fine_tuning' requires `ft_epochs` to be set.\n"
            "  - For fine-tuning from a checkpoint (Case C): set both ft_epochs and checkpoint_name.\n"
            "  - For fine-tuning a freshly created model (Case D): set `ft_epochs`."
        )

    # 5. best_model_metric must be a valid metric
    valid_metrics = (
        "recall",
        "precision",
        "auroc",
        "f1_score",
        "specificity",
        "accuracy",
    )
    if best_model_metric not in valid_metrics:
        raise ValueError(
            f"`best_model_metric` must be one of {valid_metrics} ('{best_model_metric}' given)."
        )

    # 6. recall_threshold must be between 0 and 1
    if not (0 <= recall_threshold <= 1):
        raise ValueError(
            f"`recall_threshold` must be between 0 and 1 (inclusive) ({recall_threshold} given)."
        )

    # Resolve persistent_workers
    if persistent_workers is None:
        persistent_workers = True if num_workers > 0 else False

    # Auto-disable multiprocessing if running in a problematic environment

    if (
        num_workers > 0
        and getattr(sys.modules.get("__main__"), "__spec__", None) is None
    ):
        warnings.warn(
            f"Detected notebook/Kaggle environment. Disabling multiprocessing (num_workers) to avoid "
            f"AttributeError: module '__main__' has no attribute '__spec__'. "
            f"Originally requested num_workers={num_workers}, but setting to 0.",
            RuntimeWarning,
            stacklevel=2,
        )
        print()
        num_workers = 0
        persistent_workers = False

    # Set the start method for multiprocessing. This is crucial for DataLoaders with num_workers > 0 in Colab.
    if persistent_workers:
        torch.multiprocessing.set_start_method("spawn", force=True)

    # DATALOADER RESOLUTION-------------------
    train_dl, val_dl, _ = create_dataloaders(
        train_val_dir=train_val_dir,
        train_transform=train_transform,
        test_transform=test_transform,
        val_size=val_size,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        dataloader_type="train",
    )

    # POS WEIGHT RESOLUTION---------------------
    pos_weight_resolved = torch.tensor(pos_weight, device=device, dtype=torch.float32)

    # LOSS FUNCTION---------------------
    loss_fn = _loss_fn(name="binary_ce", pos_weight=pos_weight_resolved)

    # Case A and B
    # feature extraction --> fine tune if ft_epochs given
    if mode == "transfer_learning":
        # Create a model
        model = create_model(model_name)

        # Create an optimizer for model
        tf_optimizer = create_optimizer(
            model_name=model_name,
            model=model,
            optimizer_name=optimizer_name,
            mode="transfer_learning",
            tf_lr=tf_lr,
        )

        return run_transfer_learning_experiment(
            model_name=model_name,
            model=model,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            tf_optimizer=tf_optimizer,
            loss_fn=loss_fn,
            epochs=epochs,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            tf_lr=tf_lr,
            ft_lr=ft_lr,
            lr_decay=lr_decay,
            n_layers=n_layers,
            ft_epochs=ft_epochs,
            extra_config=extra_config,
            artifacts_dir=artifacts_dir,
            project_name=project_name,
            device=device,
            best_model_metric=best_model_metric,
            recall_threshold=recall_threshold,
            extra_wandb_tags=extra_wandb_tags,
        )

    # Case C and D
    # fine tuning from existing checkpoint or standalone fine tune

    if checkpoint_name is not None:
        # Case C: finetune from existing checkpoint

        checkpoint_stem = Path(checkpoint_name).stem
        checkpoint_path = _model_artifacts_dir / f"{checkpoint_stem}.pth"
        checkpoint_path = _resolve_checkpoint_path(
            checkpoint_stem=checkpoint_stem,
            checkpoint_path=checkpoint_path,
            project_name=project_name,
        )

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        tl_run_name = checkpoint.get("run_name", "unknown-tl-run")
        tl_model_name = checkpoint.get("model_name", model_name)

        # 5. model name mismatch guard
        if tl_model_name != model_name:
            raise ValueError(
                f"Model name mismatch: given '{model_name}' but the checkpoint at '{checkpoint_path}' was saved from '{tl_model_name}'"
            )

        model = create_model(model_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"\n[INFO] Loaded checkpoint from '{checkpoint_path}'")
        print(
            f"[INFO] Parent TL run : '{tl_run_name}' "
            f"(epoch {checkpoint['epoch']}, AU-ROC: {checkpoint['best_auc']:.4f})"
        )

        tl_checkpoint_path = checkpoint_path
        from_checkpoint = True

    else:
        # Case D: standalone finetune
        model = create_model(model_name)
        tl_run_name = None
        tl_checkpoint_path = None
        from_checkpoint = False

    # unfreeze backbone blocks (Cases C and D both)
    unfreeze_for_finetune(model_name, model, n_layers)
    print(f"[INFO] Unfroze last {n_layers} backbone block(s) for fine-tuning.")

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

    return run_fine_tuning_experiment(
        model_name=model_name,
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        ft_optimizer=ft_optimizer,
        loss_fn=loss_fn,
        ft_epochs=ft_epochs,
        tf_lr=tf_lr,
        ft_lr=ft_lr,
        n_layers=n_layers,
        batch_size=batch_size,
        tl_run_name=tl_run_name,
        tl_checkpoint_path=tl_checkpoint_path,
        from_checkpoint=from_checkpoint,
        extra_config=extra_config,
        artifacts_dir=artifacts_dir,
        project_name=project_name,
        device=device,
        best_model_metric=best_model_metric,
        recall_threshold=recall_threshold,
        extra_wandb_tags=extra_wandb_tags,
    )


## Helper functions ---------------------


def _loss_fn(name="binary_ce", pos_weight: Optional[float] = None):
    """
    Creates and returns the loss function based on the given name.

    Raises:
        - ValueError if name is not a supported loss function.
    """

    if name == "binary_ce":
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    raise ValueError(f"Unknown loss function: '{name}'. Available: ['binary_ce']")


def _resolve_checkpoint_path(
    checkpoint_stem: str,
    checkpoint_path: Path,
    project_name: str,
) -> Path:
    """
    Ensures the checkpoint .pth file is available locally and returns its path.

    If the file already exists at 'load_checkpoint' it is returned as-is.
    Otherwise the matching W&B artifact (name='checkpoint_stem', type='model')
    is downloaded from 'project_name' into 'artifacts/models/' and moved to
    the canonical flat path 'artifacts/models/<checkpoint_stem>.pth'.

    NOTE: Here W&B API is used to prevent creating a run (which is not necessary here)

    Args:
        - checkpoint_stem   : bare run-name without extension, e.g. 'vit_b_16-TL_LR1e-4-EP5-B32'
        - checkpoint_path   : expected local path (artifacts/models/<stem>.pth)
        - project_name      : W&B project name used to look up the artifact

    Returns:
        - Path to the local .pth file, guaranteed to exist.

    Raises:
        - FileNotFoundError : file is absent locally AND the W&B download fails
        - FileNotFoundError : downloaded artifact directory contains no .pth file
    """

    if checkpoint_path.exists():
        return checkpoint_path

    print(
        f"[INFO] Checkpoint '{checkpoint_path}' not found locally. "
        f"Attempting to download artifact '{checkpoint_stem}' from W&B project '{project_name}'…"
    )

    _api = wandb.Api()
    try:
        artifact = _api.artifact(
            f"{project_name}/{checkpoint_stem}:latest", type="model"
        )
        # W&B always downloads into a sub-directory; stage it next to the target.
        artifact_dir = Path(
            artifact.download(root=str(checkpoint_path.parent / checkpoint_stem))
        )
    except wandb.errors.CommError as exc:
        raise FileNotFoundError(
            f"Could not find checkpoint locally at '{checkpoint_path}' and failed to "
            f"download artifact '{checkpoint_stem}:latest' from W&B project '{project_name}'.\n"
            f"W&B error: {exc}"
        ) from exc

    # The artifact was logged with add_file(checkpoint_path) so the .pth file
    # sits directly inside the downloaded directory.
    downloaded_pth = next(artifact_dir.glob("*.pth"), None)
    if downloaded_pth is None:
        raise FileNotFoundError(
            f"No .pth file found in the downloaded W&B artifact dir '{artifact_dir}'."
        )

    # Move to the canonical flat location: artifacts/models/<stem>.pth
    shutil.move(str(downloaded_pth), str(checkpoint_path))

    # Clean up the now-empty staging sub-directory
    try:
        artifact_dir.rmdir()
    except OSError:
        pass  # not empty – leave it

    print(f"[INFO] Artifact downloaded and saved to '{checkpoint_path}'.")
    return checkpoint_path
