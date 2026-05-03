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
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

from ..dataloader import create_dataloaders, get_class_weights
from ..models.models import create_model, create_optimizer, unfreeze_for_finetune
from ..transform import test_transforms, train_transforms
from ..utils import set_seeds
from .fine_tuning_experiment import run_fine_tuning_experiment
from .transfer_learning_experiment import run_transfer_learning_experiment

set_seeds()


# NOTE: later update this constants from .yaml file.
_project_name = "pneumonia-detection"

_train_val_dir = "/content/drive/MyDrive/Colab Notebooks/My Projects/Pneumonia Detection/data/chest_xray/train"
_artifacts_dir = "/content/drive/MyDrive/Colab Notebooks/My Projects/Pneumonia Detection/artifacts/models"

_batch_size = 32
_val_size = 0.2
_epochs = 5

_device = "cuda" if torch.cuda.is_available() else "cpu"
_num_workers = os.cpu_count()

_tf_lr = 1e-3
_ft_lr = 1e-5
_lr_decay = 0.2
_n_layers = 1

_pos_weight = 0.673


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
    persistent_workers: Optional[bool] = None,
    # loss fn
    pos_weight: float = _pos_weight,
    # training
    epochs: int = _epochs,
    artifacts_dir: str = _artifacts_dir,
    device: str = _device,
    # optimizer/model architecture
    tf_lr: float = _tf_lr,
    ft_lr: float = _ft_lr,
    lr_decay: float = _lr_decay,
    n_layers: int = _n_layers,
    # fine tune control
    ft_epochs: Optional[int] = None,
    load_checkpoint: Optional[str] = None,
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
    Case C │ mode="fine_tuning", load_checkpoint=<path>, ft_epochs=<int>
           │ Fine-tuning from existing checkpoint.
           │ Returns ft_results.
           │
    Case D │ mode="fine_tuning", load_checkpoint=None, ft_epochs=<int>
           │ Fine-tuning only on a freshly created model.
           │ Returns ft_results.

    Args:
        - model_name: name of the model to be used. Eg: resent50 (must be present in model_registry)
        - mode: "transfer_learning" or "fine_tuning"

        - project_name: W&B project name
        - extra_config: extra config to be passed to W&B

        - epochs: number of epochs to train for furture extraction
        - artifacts_dir: directory where the best checkpoint .pth are saved
        - device: device to train on

        - tf_lr: transfer learning learning rate
        - ft_lr: fine tuning learning rate
        - lr_decay: learning rate decay factor
        - n_layers: number of layers to unfreeze for fine tuning

        - ft_epochs: number of epochs to train for fine tuning
        - load_checkpoint: path to the checkpoint to load for fine tuning. Eg: '/content/drive/MyDrive/Colab Notebooks/My Projects/Pneumonia Detection/artifacts/models/resnet50-TL_LR1e-3-EP8-B32'

    Raises:
        - ValueError : mode is not "transfer_learning" or "fine_tuning"
        - ValueError : mode="transfer_learning" and load_checkpoint is given
        - ValueError : mode="fine_tuning" and ft_epochs = None
        - ValueError : load_checkpoint given and model_name does not match saved model
    """

    # Resolve persistent_workers
    if persistent_workers is None:
        persistent_workers = True if num_workers > 0 else False

    # Set the start method for multiprocessing. This is crucial for DataLoaders with num_workers > 0 in Colab.
    if persistent_workers:
        torch.multiprocessing.set_start_method("spawn", force=True)

    artifacts_dir = Path(artifacts_dir)
    train_val_dir = Path(train_val_dir)
    load_checkpoint = Path(load_checkpoint) if load_checkpoint else None

    # VALIDATE PARAMS---------------------

    # 1. Invalid mode
    if mode not in ("transfer_learning", "fine_tuning"):
        raise ValueError(
            f"`mode` must be 'transfer_learning' or 'fine_tuning' ('{mode}' given)."
        )

    # 2. load_checkpoint is meaningless for transfer learning
    if mode == "transfer_learning" and load_checkpoint is not None:
        raise ValueError(
            "`load_checkpoint` cannot be used with mode='transfer_learning'. "
            "To fine-tune an existing checkpoint, use mode='fine_tuning'."
        )

    # 3. fine_tuning requires at least one of load_checkpoint or ft_epochs
    if mode == "fine_tuning" and ft_epochs is None:
        raise ValueError(
            "mode='fine_tuning' requires `ft_epochs` to be set.\n"
            "  - For fine-tuning from a checkpoint (Case C): set both ft_epochs and load_checkpoint.\n"
            "  - For fine-tuning a freshly created model (Case D): set `ft_epochs`."
        )

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
        # 1. Create a model
        model = create_model(model_name)

        # 2. Create an optimizer for model
        tf_optimizer = create_optimizer(
            model_name=model_name,
            model=model,
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
            tf_lr=tf_lr,
            ft_lr=ft_lr,
            lr_decay=lr_decay,
            n_layers=n_layers,
            ft_epochs=ft_epochs,
            extra_config=extra_config,
            artifacts_dir=artifacts_dir,
            project_name=project_name,
            device=device,
        )

    # Case C and D
    # fine tuning from existing checkpoint or standalone fine tune

    if load_checkpoint is not None:
        # Case C: finetune from existing checkpoint

        checkpoint = torch.load(load_checkpoint, map_location=device, weights_only=True)

        tl_run_name = checkpoint.get("run_name", "unknown-tl-run")
        tl_model_name = checkpoint.get("model_name", model_name)

        # 4. model name mismatch guard
        if tl_model_name != model_name:
            raise ValueError(
                f"Model name mismatch: given '{model_name}' but the checkpoint at '{load_checkpoint}' was saved from '{tl_model_name}'"
            )

        model = create_model(model_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"\n[INFO] Loaded checkpoint from '{load_checkpoint}'")
        print(
            f"[INFO] Parent TL run : '{tl_run_name}' "
            f"(epoch {checkpoint['epoch']}, AU-ROC: {checkpoint['best_auc']:.4f})"
        )

        tl_checkpoint_path = load_checkpoint
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
    )


def _loss_fn(name="binary_ce", pos_weight: Optional[float] = None):
    """
    Creates and returns the loss function based on the given name.

    Raises:
        - ValueError if name is not a supported loss function.
    """

    if name == "binary_ce":
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    raise ValueError(f"Unknown loss function: '{name}'. Available: ['binary_ce']")
