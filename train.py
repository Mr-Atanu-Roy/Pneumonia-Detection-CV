"""
CLI entry point for training pneumonia detection models.
Calls the run_experiment() function in experiment.py, which handles all training logic and supports 4 main experiment types.

Usage examples:
    # Transfer learning with all defaults from config.yaml (Case A)
    python train.py --model resnet50

    # Transfer learning with custom epochs and learning rate (Case A)
    python train.py --model densenet121 --epochs 10 --tf-lr 1e-4

    # Transfer learning followed by fine-tuning (Case B)
    python train.py --model resnet50 --ft-epochs 5 --n-layers 2

    # Fine-tune from an existing checkpoint (Case C)
    python train.py --model resnet50 --mode fine_tuning --ft-epochs 5 --load-checkpoint path/to/checkpoint.pth

    # Standalone fine-tuning on a fresh model (Case D)
    python train.py --model vit_b_16 --mode fine_tuning --ft-epochs 8 --n-layers 3

    # Passing additional W&B tags
    python train.py --model efficientnet_b2 --wandb-tags baseline test-run
"""

import argparse

from src.utils import load_config


def _int_or_auto(value: str) -> int | str:
    """
    Accepts either an integer string ("4") or the literal "auto".
    Used for arguments that can be auto-resolved at runtime.
    Used for num_workers and persistent_workers.

    Args:
        value: raw string from argparse

    Returns:
        int if value is a valid integer, "auto" if value is "auto"

    Raises:
        argparse.ArgumentTypeError: if value is neither an int nor "auto"
    """
    if value == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid value '{value}'. Expected an integer or 'auto'."
        )


def parse_args(cfg: dict) -> argparse.Namespace:
    """
    Parse CLI arguments. Defaults are loaded from config.yaml

    Args:
        - cfg: config dictionary loaded from config.yaml

    Returns:
        - argparse.Namespace: parsed CLI arguments

    """

    # Load defaults from config.yaml
    c_wandb = cfg["wandb"]
    c_data = cfg["data"]
    c_dl = cfg["dataloader"]
    c_training = cfg["training"]
    c_opt = cfg["optimizer"]
    c_model = cfg["model"]
    c_paths = cfg["paths"]

    # Argument parser initialization
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train pneumonia detection models with transfer learning or fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # to shows defaults in --help
    )

    ## Adding CLI arguments with defaults from config.yaml-----------

    # required arg
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["resnet50", "densenet121", "efficientnet_b2", "vit_b_16"],
        help="Model architecture to train.",
    )

    # optional args with defaults

    parser.add_argument(
        "--mode",
        type=str,
        default="transfer_learning",
        choices=["transfer_learning", "fine_tuning"],
        help="Experiment training mode.",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=c_wandb["project_name"],
        help="W&B project name for logging.",
    )

    parser.add_argument(
        "--train-val-dir",
        type=str,
        default=c_paths["train_val_dir"],
        help="Path to the train/val data directory.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=c_paths["artifacts_dir"],
        help="Path to the artifacts directory.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=c_dl["batch_size"],
        help="Batch size for dataloaders.",
    )
    parser.add_argument(
        "--num-workers",
        type=_int_or_auto,
        default=c_dl["num_workers"]
        or "auto",  # auto is resolved in main function later
        help="Number of dataloader worker processes.",
    )
    parser.add_argument(
        "--persistent-workers",
        type=_int_or_auto,
        default=c_dl["persistent_workers"]
        or "auto",  # auto is resolved in main function later
        help="Whether to use persistent workers in dataloaders (set True if num_workers > 0, else False).",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=c_data["val_size"],
        help="Fraction of training data to use for validation.",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=c_data["pos_weight"],
        help="Positive samples weightage in train data (Used in loss function)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=c_training["epochs"],
        help="Number of epochs for transfer learning phase.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=c_training["device"]
        or "auto",  # auto is resolved in main function later
        help="Device to train on. 'auto' resolves to cuda if available, else cpu.",
    )

    parser.add_argument(
        "--tf-lr",
        type=float,
        default=c_opt["tf_lr"],
        help="Learning rate for transfer learning (classifier layer only).",
    )
    parser.add_argument(
        "--ft-lr",
        type=float,
        default=c_opt["ft_lr"],
        help="Base learning rate for fine-tuning backbone blocks.",
    )
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=c_opt["lr_decay"],
        help="Multiplicative LR decay per backbone block during fine-tuning (for discriminative learning strategy).",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=c_opt["n_layers"],
        help="Number of backbone blocks to unfreeze for fine-tuning.",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=c_model["num_classes"],
        help="Number of classes for the classification task. 1 for binary classification, >2 for multi-class classification.",
    )

    parser.add_argument(
        "--ft-epochs",
        type=int,
        default=None,
        help=(
            "Number of fine-tuning epochs. "
            "If set with --mode transfer_learning: runs TL then FT (Case B). "
            "If set with --mode fine_tuning: runs FT only (Case C or D)."
        ),
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to a .pth checkpoint to resume fine-tuning from (Case C).",
    )

    # Optional W&B tags
    parser.add_argument(
        "--wandb-tags",
        nargs="+",
        default=[],
        help="Optional space-separated W&B tags to add to the run.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point.
    - Loads config
    - parse args
    - calls run_experiment() with parsed args
    """

    cfg = load_config()
    args = parse_args(cfg)

    # Resolve 'auto' values for num_workers, persistent_workers, and device based on system capabilities

    if args.num_workers in ("auto", None):
        import os

        args.num_workers = os.cpu_count()  # set to number of CPU cores

    if args.persistent_workers in ("auto", None):
        args.persistent_workers = (
            args.num_workers > 0
        )  # True if num_workers > 0, else False

    if args.device in ("auto", None):
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Using device: {args.device}, num_workers: {args.num_workers}, persistent_workers: {args.persistent_workers}...."
    )

    from src.trainer.experiment import run_experiment

    run_experiment(
        model_name=args.model,
        mode=args.mode,
        project_name=args.project_name,
        train_val_dir=args.train_val_dir,
        batch_size=args.batch_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        pos_weight=args.pos_weight,
        epochs=args.epochs,
        artifacts_dir=args.artifacts_dir
        if args.artifacts_dir.endswith("/models")
        else f"{args.artifacts_dir}/models",
        device=args.device,
        tf_lr=args.tf_lr,
        ft_lr=args.ft_lr,
        lr_decay=args.lr_decay,
        n_layers=args.n_layers,
        ft_epochs=args.ft_epochs,
        load_checkpoint=args.load_checkpoint,
        extra_wandb_tags=args.wandb_tags,
    )


if __name__ == "__main__":
    main()
