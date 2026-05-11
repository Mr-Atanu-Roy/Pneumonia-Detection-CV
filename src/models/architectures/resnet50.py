"""
Contains functions to do the following for resnet50 model:
    - create_model: instantiate a resnet50 model for transfer learning with pretrained weights (trained on IMAGENET1K_V2 dataset). Freeze the backbone and keeps the classifier layer trainable.

    - create_optimizer: create optimizer for transfer learning or fine-tuning

    - unfreeze_for_finetune: unfreeze the last N residual blocks for fine tuning the model.
      Classifier layer is kept unfrozen. Keeps remaining layers frozen.

"""

import torch
from torchvision import models

# Ordered from last to first (unfreezing starts from the end)
_RESNET_BLOCKS = ["layer4", "layer3", "layer2", "layer1"]

_MAX_LAYERS = len(_RESNET_BLOCKS)


def _validate_n_layers(n_layers: int) -> None:
    """
    Validates that n_layers is within the acceptable range [1, _MAX_LAYERS].

    Args:
        - n_layers (int): number of residual blocks to unfreeze

    Raises:
        - ValueError: if n_layers is out of range
    """
    if n_layers < 1 or n_layers > _MAX_LAYERS:
        raise ValueError(
            f"n_layers must be between 1 and {_MAX_LAYERS} for ResNet50. "
            f"({n_layers} given)"
        )


def create_model(num_classes: int = 2) -> torch.nn.Module:
    """
    Creates a resnet50 model with pretrained weights (trained on IMAGENET1K_V2 dataset) and freezes the backbone.
    Makes the classifier layer trainable with num_classes output nodes.
    Useful for transfer learning.

    Args:
        - num_classes (int): number of output classes

    Returns:
        - model (torch.nn.Module): resnet50
    """

    # 1. Initialize a resnet50 model pretrained on IMAGENET1K_V2 dataset
    model_weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=model_weights)

    # 2. freeze the all layers
    for param in model.parameters():
        param.requires_grad = False

    # 3. Update the classifier (fc) layer to make last layer o/p as num_classes (trainable by default)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features, out_features=num_classes, bias=True
    )

    return model


def create_optimizer(
    model: torch.nn.Module,
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
    mode: str = "transfer_learning",
    n_layers: int = 1,
    tf_lr: float = 1e-3,
    ft_lr: float = 1e-5,
    lr_decay: float = 0.1,
) -> torch.optim.Optimizer:
    """
    Creates optimizer for transfer learning or fine-tuning.
    For transfer learning: only the fc layer parameters are updated.
    For fine-tuning: discriminative learning rates are applied across residual blocks.
    Each earlier block is scaled down by lr_decay:
        layer4  (position 0) -> ft_lr * (lr_decay ** 0) = ft_lr
        layer3  (position 1) -> ft_lr * (lr_decay ** 1)
        layer2  (position 2) -> ft_lr * (lr_decay ** 2)
        layer1  (position 3) -> ft_lr * (lr_decay ** 3)

    BN layers are internal to each residual block and are included automatically.
    The fc layer always uses tf_lr, independent of the decay chain.

    When n_layers=1, lr_decay has no effect (single block, no decay to apply).
    Args:
        - model (torch.nn.Module): model to be optimized
        - mode (str): "transfer_learning" or "fine_tuning"
        - tf_lr (float): learning rate for the fc layer (independent of decay chain)
        - ft_lr (float): base learning rate for the last unfrozen residual block (layer4)
        - n_layers (int): number of residual blocks to include in optimizer (fine_tuning only). Must be between 1 and 4.
        - lr_decay (float): multiplicative decay applied per block going deeper into the backbone. Must be in (0.0, 1.0]. Default 0.1. Typical values: 0.1 (aggressive), 0.3 (moderate).

    Returns:
        - optimizer (torch.optim.Optimizer)
    """

    # make sure mode is either transfer_learning or fine_tuning
    if mode not in ["transfer_learning", "fine_tuning"]:
        raise ValueError(
            f"Mode must be either 'transfer_learning' or 'fine_tuning'. ({mode} given)"
        )

    # Return optimizer for transfer_learning
    if mode == "transfer_learning":
        return optimizer_cls(params=model.fc.parameters(), lr=tf_lr)

    # for fine tuning
    _validate_n_layers(n_layers)

    param_groups = []

    # residual blocks params
    for i, residual_block in enumerate(_RESNET_BLOCKS[:n_layers]):
        block_lr = ft_lr * (lr_decay**i)
        param_groups.append(
            {"params": getattr(model, residual_block).parameters(), "lr": block_lr}
        )

    return optimizer_cls(param_groups)


def unfreeze_for_finetune(model: torch.nn.Module, n_layers: int = 1) -> None:
    """
    Unfreezes the last n_layers residual blocks of resent50.
    NOTE: the classifier layer (fc) is already unfrozen.

    Block unfreezing order (last to first):
        n_layers=1: layer4
        n_layers=2: layer4, layer3
        n_layers=3: layer4, layer3, layer2
        n_layers=4: layer4, layer3, layer2, layer1

    Args:
        - model (torch.nn.Module): resnet50 model
        - n_layers (int): number of residual blocks to unfreeze. Must be between 1 and 4.

    Raises:
        - ValueError: if n_layers is out of range [1, 4]
    """

    _validate_n_layers(n_layers)

    for residual_block in _RESNET_BLOCKS[:n_layers]:
        for param in getattr(model, residual_block).parameters():
            param.requires_grad = True

    # return is not required as nn.Module objs are mutable
