import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seeds(seed: int = 42) -> None:
    """
    Sets seed across all libraries for full reproducibility.
    Args:
        - seed (int): seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from collections import Counter


def get_loader_distribution(loader: torch.utils.data.DataLoader) -> Counter:
    """
    Returns the distribution of data in the dataloader

    Args:
        - loader (torch.utils.data.DataLoader): dataloader to get the distribution of

    Returns:
        - dict
    """

    counter = Counter()

    for _, labels in loader:
        counter.update(labels.tolist())

    return counter


import wandb


def wandb_login(platform: str) -> None:
    """
    Logs into WandB using the API key stored in the environment or secrets manager, depending on the platform.

    Args:
        platform : "colab" or "kaggle"
    """
    if platform == "colab":
        from google.colab import userdata

        api_key = userdata.get("WANDB_API_KEY")

    elif platform == "kaggle":
        from kaggle_secrets import UserSecretsClient

        api_key = UserSecretsClient().get_secret("WANDB_API_KEY")

    else:
        raise ValueError(f"Unsupported platform: {platform}")

    wandb.login(key=api_key)

    print("[INFO] WandB login successful.")


def fmt_lr(lr: float) -> str:
    """Formats 0.001 → '1e-3', 0.0001 → '1e-4', 0.1 → '1e-1'."""

    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")


def build_tl_run_name(
    model_name: str, epochs: int, tf_lr: float, batch_size: int
) -> str:
    """
    Builds run name for the Transfer Learning phase.
    Used as-is for Case A, and as the base for Case B FT name.

    Example:
        resnet50-TL_LR1e-3-EP8-B32
    """
    return f"{model_name.lower()}-TL_LR{fmt_lr(tf_lr)}-EP{epochs}-B{batch_size}"


def build_ft_suffix(ft_lr: float, ft_epochs: int, n_layers: int) -> str:
    """
    Builds the fine-tuning suffix appended to a TL run name.
    Used for Cases B and C.

    Example:
        __FT_LR1e-5-EP5-N_LY2
    """
    return f"__FT_LR{fmt_lr(ft_lr)}-EP{ft_epochs}-N_LY{n_layers}"


def build_standalone_ft_run_name(
    model_name: str, ft_lr: float, ft_epochs: int, n_layers: int
) -> str:
    """
    Builds run name for Case D — fine-tuning with no prior TL phase.

    Example:
        resnet50-FT_LR1e-5-EP5-N_LY2
    """
    return f"{model_name.lower()}-FT_LR{fmt_lr(ft_lr)}-EP{ft_epochs}-N_LY{n_layers}"


def build_config(
    model_name: str,
    mode: str,
    epochs: int,
    batch_size: int,
    tf_lr: float,
    ft_lr: Optional[float],
    n_layers: Optional[int],
    ft_epochs: Optional[int],
    load_checkpoint: Optional[str],
    extra: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Builds the base W&B config dict capturing all hyperparameters.
    Additional phase-specific keys are added by each _run_* function.
    """

    config = {
        "model_name": model_name,
        "mode": mode,
        "epochs": epochs,
        "batch_size": batch_size,
        "tf_lr": tf_lr,
        "ft_lr": ft_lr,
        "n_layers": n_layers,
        "ft_epochs": ft_epochs,
        "load_checkpoint": load_checkpoint,
    }
    if extra:
        config.update(extra)
    return config
