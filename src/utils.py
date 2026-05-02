import random

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


def get_loader_distribution(loader: torch.utils.data.DataLoader):
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


# ── Cell 3: WandB login ─────────────────────────────────────────────
import wandb


def wandb_login(platform: str) -> None:
    """
    Logs into WandB in a platform-appropriate way.

    On Colab : interactive browser-based login (wandb.login() with no args)
    On Kaggle: reads API key from Kaggle Secrets

    Args:
        platform : "colab" or "kaggle"
    """
    if platform == "colab":
        wandb.login()

    elif platform == "kaggle":
        from kaggle_secrets import UserSecretsClient

        api_key = UserSecretsClient().get_secret("WANDB_API_KEY")
        wandb.login(key=api_key)

    print("[INFO] WandB login successful.")
