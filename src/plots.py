
import torch
import numpy as np
from .transform import IMAGENET_STD, IMAGENET_MEAN

def denormalize(tensor: torch.Tensor)->np.ndarray:
    """
    Denormalize a tensor of shape (C, H, W) and converts to numpy array of shape (H, W, C). Returns a numpy array suitable for plotting.

    Normalization: (input - mean) / std
    Denormalization: input = (output × std) + mean

    Args:
        - tensor (torch.Tensor): tensor to be denormalized

    Returns:
        - np.ndarray: denormalized tensor

    """

    #make the STD & MEAN tensor obj
    std = torch.tensor(IMAGENET_STD)
    mean = torch.tensor(IMAGENET_MEAN)

    #clone to avoide modifying original tensor
    image = tensor.clone()

    #denormalize: (image * std) + mean
    image = image * std.view(3, 1, 1) + mean.view(3, 1, 1)

    image = image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    return image


import matplotlib.pyplot as plt
import random
from typing import List

def plot_data(dataloader: torch.utils.data.DataLoader,
              class_names: List[str],
              k: int=10,
              title: str="Random Sample From Images")->None:

    """
    Plots k samples for the given dataloader

    Args:
        - dataloader (torch.utils.data.DataLoader): dataloader to plot
        - k (int): number of samples to plot

    Raise:
        - ValueError: if k is not a multiple of 2 or greater than 10
    """

    # k must be multiple of 2 and should be <= 10
    if k % 2 != 0 or k > 10:
        raise ValueError("k must be a multiple of 2 and should be <= 10")

    img, labels = next(iter(dataloader))

    #get k random samples from the datasetloader
    random_idx = random.sample(range(len(img)), k)


    #plot the samples
    fig, axes = plt.subplots(nrows=2, ncols=k//2, figsize=(12, 7))
    axes = axes.flatten()  #flatten to 1D array

    for i, idx in enumerate(random_idx):

        axes[i].imshow(denormalize(img[idx]))   #denormalize before plotting

        axes[i].set_title(f"{class_names[labels[idx]]}")
        axes[i].axis("off")


    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.show()

