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


def build_wandb_config(
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


from pathlib import Path

import yaml


def load_config() -> dict:
    """
    Loads config.yaml from the repo root.
    Resolves the path relative to this file's location so it works
    regardless of where the script is called from.

    src/utils.py → src/ → repo root

    Returns:
        dict: parsed config dictionary

    Raises:
        FileNotFoundError: if config.yaml is not found at the repo root
    """
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"[ERROR] config.yaml not found at {config_path}.\n"
            f"        Ensure config.yaml exists at the repo root"
        )

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def update_yaml_config(config_path: str, updates: Dict[str, Any]) -> None:
    """
    Updates a YAML config file with the provided dictionary of updates.

    Args:
        - config_path (str): path to the YAML config file to be updated
        - updates (Dict[str, Any]): dictionary containing the keys and values to update in the config file

    Raises:
        ValueError: if the config file is empty or contains invalid YAML content.

    Example usage:
        update_yaml_config(
            config_path = REPO_DIR / "config.yaml",
            updates = {
                "training": {
                    "device": device
                },
                "dataloader": {
                    "num_workers": os.cpu_count() if device == "cuda" else 0
                },
                "paths": {
                    "train_val_dir": str(train_val_dir),
                    "test_dir": str(test_dir),
                    "artifacts_dir": str(model_artifacts_dir)
                }
            }
        )
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # if the config file is empty, raise an error
    if config is None:
        raise ValueError(
            f"Config file at {config_path} is empty. Cannot update empty files with {updates}."
        )

    _deep_update(config, updates)

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    print("[INFO] Config file updated successfully.")


def _deep_update(original: dict, updates: dict) -> None:
    """
    Recursively updates a nested dictionary with another dictionary.
    """

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(original.get(key), dict):
            _deep_update(original[key], value)
        else:
            original[key] = value
