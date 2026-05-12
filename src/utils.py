import random
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
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


def _save_single_df_and_log(df: pd.DataFrame, name: str, output_dir: Path) -> str:
    """
    Helper function to save a single DataFrame to CSV and log it as a W&B table.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] DataFrame saved to {csv_path}")
    wandb.log({name: wandb.Table(dataframe=df)})
    return str(csv_path)


def save_wandb_table(
    df: Union[pd.DataFrame, List[pd.DataFrame]],
    file_name: Union[str, List[str]],
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
    save_as_artifact: bool = True,
) -> Union[str, List[str]]:
    """
    Saves one or more pd.DataFrame objects to CSV files, logs them as W&B tables
    under a single W&B run, and optionally uploads them together as a single
    W&B artifact.

    Args:
        - df (pd.DataFrame or list[pd.DataFrame]): DataFrame(s) to save and log
        - file_name (str or list[str]): Name(s) for the CSV file(s). If a single
          string is provided it will be used for the single DataFrame. Length of
          lists for `df` and `file_name` must match.
        - save_as_artifact (bool): Whether to log the file(s) as a W&B artifact
          (default: True)
        - wandb_run_name (str, optional): Name to use for the W&B run. If not
          provided the first file name is used.
        - wandb_tags (list[str], optional): Tags to attach to the W&B run.

    Returns:
        - str or list[str]: Local path(s) to the saved CSV file(s). Returns a
          single string when a single DataFrame was provided, otherwise a list.

    Raises:
        - ValueError: If file_name is empty or if lengths of provided lists don't
          match.
    """

    # Normalize inputs to lists
    if isinstance(df, pd.DataFrame):
        dfs = [df]
    elif isinstance(df, list):
        dfs = df
    else:
        raise TypeError(f"df must be a DataFrame or list of DataFrames. Got {type(df)}")

    if isinstance(file_name, str):
        names = [file_name]
    elif isinstance(file_name, list):
        names = file_name
    else:
        raise TypeError(
            f"file_name must be a string or list of strings. Got {type(file_name)}"
        )

    if len(dfs) != len(names):
        raise ValueError(
            f"Number of DataFrames ({len(dfs)}) must match number of file names ({len(names)})"
        )

    # Validate and strip .csv extension if present
    cleaned_names: List[str] = []
    for n in names:
        if not n:
            raise ValueError(f"file_name cannot be empty. Given: {names}")
        cleaned_names.append(n[:-4] if n.endswith(".csv") else n)

    # W&B init (single run for all tables)
    cfg = load_config()
    results_artifacts_dir = Path(cfg["paths"]["artifacts_dir"]) / "results"
    project_name = cfg["wandb"]["project_name"]
    run_name = wandb_run_name if wandb_run_name else cleaned_names[0]
    wandb.init(project=project_name, name=run_name, tags=wandb_tags)

    try:
        csv_paths: List[str] = []
        for df_item, name in zip(dfs, cleaned_names):
            if not isinstance(df_item, pd.DataFrame):
                raise TypeError("All items in df list must be pandas DataFrame objects")
            csv_paths.append(
                _save_single_df_and_log(df_item, name, results_artifacts_dir)
            )

        # Optionally log all files as a single artifact
        if save_as_artifact:
            artifact_name = run_name
            artifact = wandb.Artifact(name=artifact_name, type="dataset")
            for path in csv_paths:
                artifact.add_file(path)
            wandb.log_artifact(artifact)
            print(
                f"[INFO] DataFrame(s) logged as W&B artifact with name: {artifact_name}"
            )

        # Return single path for single input to preserve backwards compatibility
        return csv_paths[0] if len(csv_paths) == 1 else csv_paths

    finally:
        wandb.finish()


def convert_results_to_df(
    training_config: Dict[str, Any],
    exclude_keys: Optional[List[str]] = None,
    save_df_as_csv: bool = True,
    csv_file_name: Optional[Union[str, List[str]]] = [
        "best_model_results",
        "all_model_results",
    ],
    wandb_run_name: Optional[str] = "Model_Results_Summary",
    wandb_tags: Optional[List[str]] = None,
    save_as_wandb_artifact: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts a results dict obtained from model training into 2 pandas DataFrame: one for best model checkpoints (eval only) and another for all training metrics (train and eval).

    Args:
        - training_config (Dict[str, Any]): Dictionary containing the raw model training config which includes the configs for training and results.
        Eg:
        {
            'batch_size': 32,
            'extra_wandb_tags': ['phase 1', 'baseline'],
            'models': {
                'resnet50': {
                    'epochs': 5,
                    'n_layers': 2,
                    'tf_lr': 0.0001,
                    'results': {
                        'total_time_sec': 377.19355332400005,
                        'best_model_metric_name': 'auroc',
                        'best_model_metric_value': 0.9390890598297119,
                        'checkpoint_path': '/kaggle/working/artifacts/models/densenet121-TL_LR1e-3-EP5-B32.pth',
                        .
                        .
                        .
                        'train':
                        [{'accuracy': 0.9321871995925903,
                        'auroc': 0.9096024632453918,
                        'f1_score': 0.9543994665145874,
                        'loss': 0.18135502731258218,
                        'precision': 0.9525641202926636,
                        'recall': 0.9562419652938843,
                        'specificity': 0.8629629611968994},
                        {'accuracy': 0.9293218851089478,
                        'auroc': 0.9270055294036865,
                        'f1_score': 0.9513797760009766,
                        'loss': 0.15081403223854123,
                        'precision': 0.9718120694160461,
                        'recall': 0.9317889213562012,
                        'specificity': 0.9222221970558167}],

                        'eval': [{'accuracy': 0.9321871995925903,
                                'auroc': 0.9096024632453918,
                                'f1_score': 0.9543994665145874,
                                'loss': 0.18135502731258218,
                                'precision': 0.9525641202926636,
                                'recall': 0.9562419652938843,
                                'specificity': 0.8629629611968994},
                                {'accuracy': 0.9293218851089478,
                                'auroc': 0.9270055294036865,
                                'f1_score': 0.9513797760009766,
                                'loss': 0.15081403223854123,
                                'precision': 0.9718120694160461,
                                'recall': 0.9317889213562012,
                                'specificity': 0.9222221970558167}],
                    }
                },
                'densenet121': {
                    .......
                },
                .
                .
                .
            }
        }
        - exclude_keys (List[str], optional): List of keys to exclude from the conversion.
        - recall_threshold (float, optional): Minimum recall value required for a model checkpoint to be considered as the best model. Default is 0.9.
        - save_df_as_csv (bool, optional): Whether to save the resulting DataFrames as CSV files
        - csv_file_name (str or list[str], optional): Name(s) for the CSV file(s) to save the DataFrames. If a single string is provided it will be used for both DataFrames. If a list is provided, it must contain two strings corresponding to the two DataFrames. Default is ["best_model_results", "all_model_results"].
        - wandb_run_name (str, optional): Name to use for the W&B run when logging the DataFrames. Default is "Model_Results_Summary".
        - wandb_tags (list[str], optional): Tags to attach to the W&B run when logging the DataFrames.
        - save_as_wandb_artifact (bool, optional): Whether to log the CSV file(s) as a W&B artifact. Default is True. To set it True the save_df_as_csv must be True.

    Returns:
        - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames: one for best model checkpoints (eval only) and another for all training metrics (train and eval).

    Raises:
        - ValueError: If save_as_wandb_artifact is True but save_df_as_csv is False
    """

    if save_as_wandb_artifact and not save_df_as_csv:
        raise ValueError(
            "save_as_wandb_artifact cannot be True if save_df_as_csv is False. To log as artifact, the DataFrame(s) must first be saved as CSV file(s)."
        )

    # get the recall threshold from the config file
    cfg = load_config()
    recall_threshold = cfg["training"]["recall_threshold"]

    # Ensure exclude_keys is a list and add default keys to exclude
    exclude_keys = exclude_keys if exclude_keys else []
    exclude_keys.extend(["wandb_tags", "extra_wandb_tags"])

    best_model_df_data = []  # DF 1: one row per model with best checkpoint info and configs

    all_model_df_data = []  # DF 2: one row per epoch per model with all train and eval metrics, configs, and checkpoint info

    for model_name in training_config["models"]:
        if "results" not in training_config["models"][model_name]:
            raise ValueError(
                f"Missing 'results' key for model '{model_name}' in training_config."
            )

        base_model_info = {
            "model_name": model_name,
        }

        # add the base info common to all models for the model (excluding the keys in exclude_keys and "results")
        for base_key in training_config:
            if (
                base_key not in exclude_keys
                and base_key != "results"
                and isinstance(training_config[base_key], (str, int, float, bool))
            ):
                base_model_info[base_key] = training_config[base_key]

        # add model specific info (excluding the keys in exclude_keys and "results"). Overwrite any base info with model-specific values if there's a key overlap.
        for model_key in training_config["models"][model_name]:
            if model_key not in exclude_keys and model_key != "results":
                base_model_info[
                    "total_epochs" if model_key == "epochs" else model_key
                ] = training_config["models"][model_name][model_key]

        # results is expected to be a dict with keys like "best_model_metric_name", "best_model_metric_value", "checkpoint_path", "train" (list of dicts), "eval" (list of dicts)
        results = training_config["models"][model_name]["results"]
        base_model_info["best_model_metric_name"] = results.get(
            "best_model_metric_name"
        )
        base_model_info["best_model_metric_value"] = results.get(
            "best_model_metric_value"
        )
        base_model_info["best_checkpoint_path"] = (
            Path(results.get("checkpoint_path")).name
            if results.get("checkpoint_path")
            else None
        )
        base_model_info["total_time_sec"] = results.get("total_time_sec")
        # total_time_sec is combined time for train and eval per epoch, so divide by number of epochs
        num_epochs = len(results.get("train", []))
        base_model_info["avg_train_time_per_epoch_sec"] = (
            results.get("total_time_sec") / num_epochs if num_epochs > 0 else None
        )

        # Creating data for DF 1: one row per epoch per model with all train and eval metrics, configs, and checkpoint info
        best_model_metric_value = -1
        for epoch, eval_res in enumerate(
            results["eval"], start=1
        ):  # start counting from 1
            current_best_metric_value = eval_res.get(
                base_model_info["best_model_metric_name"]
            )
            if (
                current_best_metric_value is not None
                and current_best_metric_value > best_model_metric_value
                and eval_res["recall"] > recall_threshold
            ):
                best_model_metric_value = current_best_metric_value
                best_result = {"best_epoch": epoch, **eval_res}

        if best_model_metric_value == -1:
            # when no epoch meets the best model criteria
            best_result = {"best_epoch": None}

        best_model_df_data.append({**base_model_info, **best_result})

        # Creating data for DF 2: one row per model with best checkpoint info and configs
        model_train_result = results.get("train", [])
        model_eval_result = results.get("eval", [])
        for epoch, (train_res, eval_res) in enumerate(
            zip(model_train_result, model_eval_result), start=1
        ):
            train_row = {f"train_{k}": v for k, v in train_res.items()}
            eval_row = {f"eval_{k}": v for k, v in eval_res.items()}
            all_model_df_data.append(
                {"epoch": epoch, **base_model_info, **train_row, **eval_row}
            )

    best_model_df = pd.DataFrame(best_model_df_data)
    all_model_df = pd.DataFrame(all_model_df_data)

    saved_paths = None
    if save_df_as_csv:
        saved_paths = save_wandb_table(
            df=[best_model_df, all_model_df],
            file_name=csv_file_name,
            wandb_tags=wandb_tags if wandb_tags else [],
            wandb_run_name=wandb_run_name,
            save_as_artifact=save_as_wandb_artifact,
        )

    return {
        "best_model_df": best_model_df,
        "all_model_df": all_model_df,
        "local_saved_paths": saved_paths,
    }
