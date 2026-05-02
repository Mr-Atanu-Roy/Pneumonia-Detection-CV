"""
Contains functions to do the following for efficientnet_b2 model:
    - create_model: instantiate a efficientnet_b2 model for transfer learning. Pretrained weights (trained on IMAGENET1K_V2 dataset). Freeze the backbone and keeps the classifier layer trainable.

    - create_optimizer: create optimizer for transfer learning or fine-tuning

    - unfreeze_for_finetune: unfreeze the last N MBConv blocks for fine tuning the model.
      features[8] (Conv2dNormActivation) is always unfrozen alongside any block. Classifier layer is kept unfrozen.
      Keeps remaining layers frozen.

"""

import torch
from torchvision import models

# Ordered from last to first (unfreezing starts from the end)
# features[8] is Conv2dNormActivation — always unfrozen for fine tuning, not counted in N
# N applies to features[7] down to features[0]
_EFFICIENTNET_BLOCKS = list(range(7, -1, -1))  # [7, 6, 5, 4, 3, 2, 1, 0]

_MAX_LAYERS = len(_EFFICIENTNET_BLOCKS)



def _validate_n_layers(n_layers: int) -> None:
    """
    Validates that n_layers is within the acceptable range [1, _MAX_LAYERS].

    Args:
        - n_layers (int): number of MBConv blocks to unfreeze

    Raises:
        - ValueError: if n_layers is out of range
    """
    if n_layers < 1 or n_layers > _MAX_LAYERS:
        raise ValueError(
            f"n_layers must be between 1 and {_MAX_LAYERS} for EfficientNet-B2. "
            f"({n_layers} given)"
        )


def create_model(num_classes: int=2)->torch.nn.Module:
    """
    Creates a efficientnet_b2 model with pretrained weights (trained on IMAGENET1K_V2 dataset) and freeze the backbone.
    Makes the classifier layer trainable with num_classes output nodes.
    Usefull for transfer learning.

    Args:
        - num_classes (int): number of output classes

    Returns:
        - model (torch.nn.Module): efficientnet_b2

    """

    # 1. Initialize a efficientnet_b2 model pretrained on IMAGENET1K_V2 dataset
    model_weights = models.EfficientNet_B2_Weights.DEFAULT
    model = models.efficientnet_b2(weights=model_weights)

    # 2. freeze the all layers
    for param in model.parameters():
        param.requires_grad = False

    # 3. Update the classifier layer to make last layer o/p as num_classes (trainable by default)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True),
        torch.nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=num_classes,
            bias=True
        )
    )

    return model


def create_optimizer(model: torch.nn.Module,
                     mode: str="transfer_learning",
                     n_layers: int=1,
                     tf_lr: float=1e-3,
                     ft_lr: float=1e-5,
                     lr_decay: float=0.1)->torch.optim.Optimizer:

    """
    Creates optimizer for transfer learning or fine-tuning.
    For transfer learning: only the classifier layer parameters are updated.
    For fine-tuning: discriminative learning rates are applied across MBConv blocks.
    features[8] (Conv2dNormActivation) sits at position i=0 (same level as features[7]) and always gets ft_lr. Each earlier MBConv block is scaled down by lr_decay:
        features[8]  (position 0) -> ft_lr * (lr_decay ** 0) = ft_lr [always]
        features[7]  (position 0) -> ft_lr * (lr_decay ** 0) = ft_lr
        features[6]  (position 1) -> ft_lr * (lr_decay ** 1)
        features[5]  (position 2) -> ft_lr * (lr_decay ** 2)
        ...
        features[0]  (position 7) -> ft_lr * (lr_decay ** 7)

    The classifier always uses tf_lr, independent of the decay chain.

    When n_layers=1, lr_decay has no effect (single block, no decay to apply).

    Args:
        - model (torch.nn.Module): model to be optimized
        - mode (str): "transfer_learning" or "fine_tuning"
        - tf_lr (float): learning rate for the classifier layer (independent of decay chain)
        - ft_lr (float): base learning rate for the last unfrozen MBConv block (features[7])
        - n_layers (int): number of MBConv blocks to include in optimizer (fine_tuning only). Must be between 1 and 8.
        - lr_decay (float): multiplicative decay applied per block going deeper into the backbone. Must be in (0.0, 1.0]. Default 0.1. Typical values: 0.1 (aggressive), 0.3 (moderate).

    Returns:
        - optimizer (torch.optim.Optimizer)
    """

    # make sure mode is eigther transfer_learning or fine_tuning
    if mode not in ["transfer_learning", "fine_tuning"]:
        raise ValueError(
            f"mode must be either 'transfer_learning' or 'fine_tuning'. ({mode} given)"
        )

    # Return optimizer for transfer_learning
    if mode=="transfer_learning":
        return torch.optim.Adam(params=model.classifier.parameters(), lr=tf_lr)


    # for fine tuning

    _validate_n_layers(n_layers)

    param_groups = []

    for i, effnet_block_idx in enumerate(_EFFICIENTNET_BLOCKS[:n_layers]):
        block_lr = ft_lr * (lr_decay ** i)
        param_groups.append(
            {"params": model.features[effnet_block_idx].parameters(), "lr": block_lr}
        )

    # Conv2dNormActivation layer (features[8]) params
    param_groups.append(
        {"params": model.features[8].parameters(), "lr": ft_lr}
    )

    # classifier head params
    param_groups.append(
        {"params": model.classifier.parameters(), "lr": tf_lr}
    )

    return torch.optim.Adam(param_groups)


def unfreeze_for_finetune(model: torch.nn.Module,
                          n_layers: int=1)->None:

    """
    Unfreezes features[8] (Conv2dNormActivation) always, plus the last n_layers
    MBConv blocks (features[7] down to features[0]).
    NOTE: the classifier layer is already unfrozen.

    Block unfreezing order (last to first):
        n_layers=1: features[8] + features[7]
        n_layers=2: features[8] + features[7], features[6]
        ...
        n_layers=8: features[8] + features[7] → features[0]

    Args:
        - model (torch.nn.Module): efficientnet_b2 model
        - n_layers (int): number of MBConv blocks to unfreeze. Must be between 1 and 8.

    Raises:
        - ValueError: if n_layers is out of range [1, 8]
    """

    # unfreeze the last n_layers MBConv blocks
    for effnet_block_idx in _EFFICIENTNET_BLOCKS[:n_layers]:
        for param in model.features[effnet_block_idx].parameters():
            param.requires_grad = True

    # unfreeze the last Conv2dNormActivation layer (features[8])
    for param in model.features[8].parameters():
        param.requires_grad = True

