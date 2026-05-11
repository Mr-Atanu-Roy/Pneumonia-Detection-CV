"""
Contains functions to do the following for vit_b_16 model:
    - create_model: instantiate a vit_b_16 model for transfer learning. Pretrained weights (trained on IMAGENET1K_V2 dataset). Freeze the backbone and keeps the classifier layer trainable.

    - create_optimizer: create optimizer for transfer learning or fine-tuning

    - unfreeze_for_finetune: unfreeze the last N encoder blocks for fine tuning the model.
      The ln layer is always unfrozen alongside any encoder block. Classifier layer is kept unfrozen.
      Keeps remaining layers frozen.

"""

import torch
from torchvision import models

# Ordered from last to first (unfreezing starts from the end)
# Total of 12 encoder layers in ViT-B/16
_ENCODER_BLOCKS = [f"encoder_layer_{i}" for i in range(11, -1, -1)]

_MAX_LAYERS = len(_ENCODER_BLOCKS)


def _validate_n_layers(n_layers: int) -> None:
    """
    Validates that n_layers is within the acceptable range [1, _MAX_LAYERS].

    Args:
        - n_layers (int): number of encoder blocks to unfreeze

    Raises:
        - ValueError: if n_layers is out of range
    """
    if n_layers < 1 or n_layers > _MAX_LAYERS:
        raise ValueError(
            f"n_layers must be between 1 and {_MAX_LAYERS} for ViT-B/16. "
            f"({n_layers} given)"
        )


def create_model(num_classes: int = 2) -> torch.nn.Module:
    """
    Creates a vit_b_16 model with pretrained weights (trained on IMAGENET1K_V2 dataset) and freezes the backbone.
    Makes the classifier layer trainable with num_classes output nodes.
    Useful for transfer learning.

    Args:
        - num_classes (int): number of output classes

    Returns:
        - model (torch.nn.Module): vit_b_16
    """

    # 1. Initialize a vit_b_16 model pretrained on IMAGENET1K_V2 dataset
    model_weights = models.ViT_B_16_Weights.DEFAULT
    model = models.vit_b_16(weights=model_weights)

    # 2. freeze the all layers
    for param in model.parameters():
        param.requires_grad = False

    # 3. Update the classifier head to make last layer o/p as num_classes (trainable by default)
    model.heads.head = torch.nn.Linear(
        in_features=model.heads.head.in_features, out_features=num_classes, bias=True
    )

    return model


def create_optimizer(
    model: torch.nn.Module,
    optimizer_cls: torch.optim.Optimizer,
    mode: str = "transfer_learning",
    n_layers: int = 1,
    tf_lr: float = 1e-4,
    ft_lr: float = 1e-5,
    lr_decay: float = 0.1,
) -> torch.optim.Optimizer:
    """
    Creates optimizer for transfer learning or fine-tuning.
    For transfer learning: only the classifier head parameters are updated.
    For fine-tuning: discriminative learning rates are applied across encoder blocks.
    ln and encoder_layer_11 are treated as one logical unit at position i=0 (both get ft_lr).
    Each earlier encoder block is scaled down by lr_decay:

        ln                  (position 0) -> ft_lr * (lr_decay ** 0) = ft_lr
        encoder_layer_11    (position 0) -> ft_lr * (lr_decay ** 0) = ft_lr
        encoder_layer_10    (position 1) -> ft_lr * (lr_decay ** 1)
        encoder_layer_9     (position 2) -> ft_lr * (lr_decay ** 2)
        ...
        encoder_layer_0    (position 11) -> ft_lr * (lr_decay ** 11)

    When n_layers=1, lr_decay has no effect (single block, no decay to apply).

    The classifier head always uses tf_lr, independent of the decay chain.

    Args:
        - model (torch.nn.Module): model to be optimized
        - optimizer_cls (torch.optim.Optimizer): optimizer class to instantiate
        - mode (str): "transfer_learning" or "fine_tuning"
        - tf_lr (float): learning rate for the classifier head (independent of decay chain)
        - ft_lr (float): base learning rate for the last unfrozen encoder block (encoder_layer_11)
        - n_layers (int): number of encoder blocks to include in optimizer (fine_tuning only). Must be between 1 and 12.
        - lr_decay (float): multiplicative decay applied per block going deeper into the encoder. Must be in (0.0, 1.0]. Default 0.1. Typical values: 0.1 (aggressive), 0.3 (moderate).
    Returns:
        - optimizer (torch.optim.Optimizer)
    """

    # make sure mode is either transfer_learning or fine_tuning
    if mode not in ["transfer_learning", "fine_tuning"]:
        raise ValueError(
            f"mode must be either 'transfer_learning' or 'fine_tuning'. ({mode} given)"
        )

    # Return optimizer for transfer_learning
    if mode == "transfer_learning":
        return optimizer_cls(params=model.heads.parameters(), lr=tf_lr)

    # for fine tuning

    _validate_n_layers(n_layers)

    param_groups = []

    # encoder layer params
    for i, encoder_layer in enumerate(_ENCODER_BLOCKS[:n_layers]):
        block_lr = ft_lr * (lr_decay**i)
        param_groups.append(
            {
                "params": getattr(model.encoder.layers, encoder_layer).parameters(),
                "lr": block_lr,
            }
        )

    # include the ln layer
    param_groups.append({"params": model.encoder.ln.parameters(), "lr": tf_lr})

    # include the classifier head params
    param_groups.append({"params": model.heads.parameters(), "lr": tf_lr})

    return optimizer_cls(param_groups)


def unfreeze_for_finetune(model: torch.nn.Module, n_layers: int = 1) -> None:
    """
    Unfreezes the last n_layers encoder blocks. The ln layer is always
    unfrozen whenever any encoder block is unfrozen.
    NOTE: the classifier head is already unfrozen.

    Block unfreezing order (last to first):
        n_layers=1 : encoder_layer_11 + ln
        n_layers=2 : encoder_layer_11, encoder_layer_10 + ln
        ...
        n_layers=12: encoder_layer_11 → encoder_layer_0 + ln

    Args:
        - model (torch.nn.Module): vit_b_16 model
        - n_layers (int): number of encoder blocks to unfreeze. Must be between 1 and 12.

    Raises:
        - ValueError: if n_layers is out of range [1, 12]
    """

    _validate_n_layers(n_layers)

    # unfreeze encoder_layers from end
    for encoder_layer in _ENCODER_BLOCKS[:n_layers]:
        for param in getattr(model.encoder.layers, encoder_layer).parameters():
            param.requires_grad = True

    # unfreeze ln
    for param in model.encoder.ln.parameters():
        param.requires_grad = True
