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
