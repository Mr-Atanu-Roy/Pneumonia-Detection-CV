"""
Contains registry of models and their architectures
"""

import torch

from .architectures import densenet121, efficientnet_b2, resnet50, vit_b_16

model_registry = {
    "resnet50": resnet50,
    "densenet121": densenet121,
    "efficientnet_b2": efficientnet_b2,
    "vit_b_16": vit_b_16,
}

optimizer_registry = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}


def _resolve_optimizer_cls(optimizer_name: str) -> type[torch.optim.Optimizer]:
    """
    Helper function to validate and return optimizer class from optimizer_registry.

    Args:
        - optimizer_name (str): name of the optimizer to validate and return
    Returns:
        - optimizer_cls (type[torch.optim.Optimizer]): optimizer class from optimizer_registry

    """

    if not isinstance(optimizer_name, str):
        raise ValueError(
            f"optimizer_name must be a string. ({type(optimizer_name)} given)"
        )

    optimizer_name = optimizer_name.lower()
    if optimizer_name not in optimizer_registry.keys():
        raise ValueError(
            f"Optimizer name: {optimizer_name} is not present in optimizer registry. "
            f"Available optimizers: {list(optimizer_registry.keys())}"
        )

    return optimizer_registry[optimizer_name]


def _validate_model(model_name: str) -> None:
    """
    helper function to validate whether a model is present in the model registry.

    Args:
        - model_name (str): name of the model to validate

    Raises:
        - ValueError: if model is not present in the registry
    """

    if not isinstance(model_name, str):
        raise ValueError(f"model_name must be a string. ({type(model_name)} given)")

    model_name = model_name.lower()
    if model_name not in model_registry.keys():
        raise ValueError(
            f"Model name: {model_name} is not present in model registry. "
            f"Available models: {list(model_registry.keys())}"
        )


def create_model(model_name: str, num_classes: int = 1) -> torch.nn.Module:
    """
    Creates a model from the registry.
    Note: By default the num_classes is set to 1 which means binary classification. For multi class classification set it to a value greater than 2.

    Args:
        - model_name (str): name of the model to create
        - num_classes (int): number of classes for o/p node of classifier layer in model

    Returns:
        - model (torch.nn.Module): model created from the registry

    Raises:
        - ValueError: if num_classes is not greater than 2 for multi class classification
    """

    _validate_model(model_name)

    # num_classes can be either 1 for binary classification or grater than 2 for multi class classification
    if num_classes == 2:
        raise ValueError(
            f"num_classes must be greater than 2 for multi class classification ({num_classes} given)"
        )

    return model_registry[model_name].create_model(num_classes=num_classes)


def create_optimizer(
    model_name: str,
    model: torch.nn.Module,
    optimizer_name: str = "adam",
    mode: str = "transfer_learning",
    n_layers: int = 1,
    tf_lr: float = 1e-3,
    ft_lr: float = 1e-5,
    lr_decay: float = 0.1,
) -> torch.optim.Optimizer:
    """
    Returns optimizer for given model_name and model instance for model present in registry.

    For transfer learning (mode="transfer_learning"): only the classifier layer is updated.

    For fine-tuning (mode="fine_tuning"): discriminative learning rates are applied across the last n_layers backbone blocks. The last block gets ft_lr, and each earlier block is scaled by lr_decay_factor:
        last block       -> ft_lr
        second last      -> ft_lr * lr_decay_factor
        third last       -> ft_lr * lr_decay_factor^2
        ...

    The classifier always uses tf_lr, independent of the decay chain.
    When lr_decay_factor=1.0 (default).

    Args:
        - model_name (str): name of the model
        - model (torch.nn.Module): model to create optimizer for
        - optimizer_name (str): optimizer to use. Must be 'adam' or 'adamw'
        - mode (str): "transfer_learning" or "fine_tuning"
        - tf_lr (float): learning rate for the classifier layer (independent of decay chain)
        - ft_lr (float): base learning rate for the last unfrozen backbone block
        - n_layers (int): number of feature blocks to optimize (fine_tuning only).
            Valid range depends on architecture:
            - resnet50       : 1 to 4
            - densenet121    : 1 to 4
            - efficientnet_b2: 1 to 8
            - vit_b_16       : 1 to 12
        - lr_decay_factor (float): multiplicative decay per block going deeper into backbone. Must be in (0.0, 1.0]. Default 0.1. Typical values: 0.1 (aggressive), 0.3 (moderate).

    Returns:
        - optimizer (torch.optim.Optimizer): optimizer for the model
    """

    _validate_model(model_name)

    optimizer_cls = _resolve_optimizer_cls(optimizer_name)

    return model_registry[model_name].create_optimizer(
        model=model,
        optimizer_cls=optimizer_cls,
        mode=mode,
        n_layers=n_layers,
        tf_lr=tf_lr,
        ft_lr=ft_lr,
        lr_decay=lr_decay,
    )


def unfreeze_for_finetune(
    model_name: str, model: torch.nn.Module, n_layers: int = 1
) -> None:
    """
    Unfreezes the last n_layers feature blocks of the given model for fine-tuning.
    Associated norm/transition layers are unfrozen alongside their respective blocks.
    NOTE: the classifier layer is already unfrozen.

    Valid n_layers range per architecture:
        - resnet50       : 1 to 4  (layer4 → layer1)
        - densenet121    : 1 to 4  (denseblock4 → denseblock1)
        - efficientnet_b2: 1 to 8  (features[7] → features[0], features[8] always unfrozen)
        - vit_b_16       : 1 to 12 (encoder_layer_11 → encoder_layer_0, ln always unfrozen)

    Args:
        - model_name (str): name of the model
        - model (torch.nn.Module): model to be fine-tuned
        - n_layers (int): number of feature blocks to unfreeze (default: 1)

    Raises:
        - ValueError: if n_layers is out of range for the given architecture
    """

    _validate_model(model_name)

    model_registry[model_name].unfreeze_for_finetune(model, n_layers=n_layers)
