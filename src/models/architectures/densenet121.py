"""
Contains functions to do the following for densenet121 model:
    - create_model: instantiate a densenet121 model for transfer learning. Pretrained weights (trained on IMAGENET1K_V2 dataset). Freeze the backbone and keeps the classifier layer trainable.

    - create_optimizer: create optimizer for transfer learning or fine-tuning

    - unfreeze_for_finetune: unfreeze the last N dense blocks for fine tuning the model.
      Classifier layer is kept unfrozen. Keeps remaining layers frozen.

"""

import torch
from torchvision import models

# Ordered from last to first (unfreezing starts from the end)
# Each entry: (dense_block_attr, associated_norm_or_transition_attr)
_DENSENET_BLOCKS = [
    ("denseblock4", "norm5"),
    ("denseblock3", "transition3"),
    ("denseblock2", "transition2"),
    ("denseblock1", "transition1"),
]
_MAX_LAYERS = len(_DENSENET_BLOCKS)


def _validate_n_layers(n_layers: int) -> None:
    """
    Validates that n_layers is within the acceptable range [1, _MAX_LAYERS].

    Args:
        - n_layers (int): number of dense blocks to unfreeze

    Raises:
        - ValueError: if n_layers is out of range
    """
    if n_layers < 1 or n_layers > _MAX_LAYERS:
        raise ValueError(
            f"n_layers must be between 1 and {_MAX_LAYERS} for DenseNet121. "
            f"({n_layers} given)"
        )


def create_model(num_classes: int = 2) -> torch.nn.Module:
    """
    Creates a densenet121 model with pretrained weights (trained on IMAGENET1K_V2 dataset) and freeze the backbone.
    Makes the classifier layer trainable with num_classes output nodes.
    Useful for transfer learning.

    Args:
        - num_classes (int): number of output classes

    Returns:
        - model (torch.nn.Module): densenet121

    """

    # 1. Initialize a densenet121 model pretrained on IMAGENET1K_V2 dataset
    model_weights = models.DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=model_weights)

    # 2. freeze the all layers
    for param in model.parameters():
        param.requires_grad = False

    # 3. Update the classifier layer to make last layer o/p as num_classes (trainable by default)
    model.classifier = torch.nn.Linear(
        in_features=model.classifier.in_features, out_features=num_classes, bias=True
    )

    return model


def create_optimizer(
    model: torch.nn.Module,
    optimizer_cls: torch.optim.Optimizer,
    mode: str = "transfer_learning",
    n_layers: int = 1,
    tf_lr: float = 1e-3,
    ft_lr: float = 1e-5,
    lr_decay: float = 0.1,
) -> torch.optim.Optimizer:
    """
    Creates optimizer for transfer learning or fine-tuning.
    For transfer learning: only the classifier layer parameters are updated.
    For fine-tuning: discriminative learning rates are applied across backbone blocks.
    The last (closest to classifier) block gets ft_lr, and each earlier block is scaled down by lr_decay:
        denseblock4  (position 0) -> ft_lr * (lr_decay ** 0) = ft_lr
        denseblock3  (position 1) -> ft_lr * (lr_decay ** 1)
        denseblock2  (position 2) -> ft_lr * (lr_decay ** 2)
        denseblock1  (position 3) -> ft_lr * (lr_decay ** 3)

    Associated norm/transition layers share the same LR as their dense block.
    The classifier always uses tf_lr, independent of the decay chain.

    When n_layers=1, lr_decay has no effect (single block, no decay to apply).

    Args:
        - model (torch.nn.Module): model to be optimized
        - optimizer_cls (torch.optim.Optimizer): optimizer class to instantiate
        - mode (str): "transfer_learning" or "fine_tuning"
        - tf_lr (float): learning rate for the classifier layer (independent of decay chain)
        - ft_lr (float): base learning rate for the last unfrozen backbone block
        - n_layers (int): number of dense blocks to include in optimizer (fine_tuning only). Must be between 1 and 4.
        - lr_decay (float): multiplicative decay applied per block going deeper into the backbone. Must be in (0.0, 1.0]. Default 0.1. Typical values: 0.1 (aggressive), 0.3 (moderate).

    Returns:
        - optimizer (torch.optim.Optimizer)
    """

    if mode not in ["transfer_learning", "fine_tuning"]:
        raise ValueError(
            f"mode must be either 'transfer_learning' or 'fine_tuning'. ({mode} given)"
        )

    # Return optimizer for transfer_learning
    if mode == "transfer_learning":
        return optimizer_cls(params=model.classifier.parameters(), lr=tf_lr)

    # for fine tuning

    _validate_n_layers(n_layers)

    param_groups = []

    # Add last n_layers dense blocks and their associated norm/transition layers
    for i, (block_attr, associated_block_attr) in enumerate(
        _DENSENET_BLOCKS[:n_layers]
    ):
        block_lr = ft_lr * (lr_decay**i)
        param_groups.append(
            {"params": getattr(model.features, block_attr).parameters(), "lr": block_lr}
        )
        param_groups.append(
            {
                "params": getattr(model.features, associated_block_attr).parameters(),
                "lr": block_lr,
            }
        )

    # include classifier head params
    param_groups.append({"params": model.classifier.parameters(), "lr": tf_lr})

    return optimizer_cls(param_groups)


def unfreeze_for_finetune(model: torch.nn.Module, n_layers: int = 1) -> None:
    """
    Unfreezes the last n_layers dense blocks and their associated norm/transition layers.
    NOTE: the classifier layer is already unfrozen.

    Block unfreezing order (last to first):
        n_layers=1: denseblock4 + norm5
        n_layers=2: denseblock4 + norm5, denseblock3 + transition3
        n_layers=3: above + denseblock2 + transition2
        n_layers=4: above + denseblock1 + transition1

    Args:
        - model (torch.nn.Module): densenet121 model
        - n_layers (int): number of dense blocks to unfreeze. Must be between 1 and 4.

    Raises:
        - ValueError: if n_layers is out of range [1, 4]
    """

    _validate_n_layers(n_layers)

    # Unfreeze last n_layers dense blocks and their associated norm/transition layers
    for block_attr, associated_block_attr in _DENSENET_BLOCKS[:n_layers]:
        # unfreeze dense block
        for param in getattr(model.features, block_attr).parameters():
            param.requires_grad = True

        # unfreeze associated block
        for param in getattr(model.features, associated_block_attr).parameters():
            param.requires_grad = True
