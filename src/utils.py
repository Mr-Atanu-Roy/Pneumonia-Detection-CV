import random
import shutil
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import wandb
import yaml

# Two artifact types officially recognised by W&B
VALID_ARTIFACT_TYPES = ("dataset", "model")


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


def is_best_model(
    best_metric_name: str,
    current_metric_value: Union[float, Tuple[float, float]],
    best_metric_value: float,
    recall: float,
    recall_threshold: float,
    weights: Tuple[float, float] = (0.7, 0.3),
) -> Dict[str, Union[bool, float]]:
    """
    Determines if the current model is the best model based on the specified metric and recall threshold.

    Args:
        - best_metric_name (str): Name of the metric used to determine the best model (e.g., "auroc"). For "composite", the metric is calculated as weighted sum Eg: 0.7 f1 + 0.3 auroc.
        - current_metric_value (float): The value of the specified metric for the current model. If best_metric_name is "composite", this should be a tuple of (f1_score, auroc_score) to calculate the composite score.
        - best_metric_value (float): The best value of the specified metric observed so far.
        - recall (float): The recall value for the current model.
        - recall_threshold (float): The minimum recall threshold required for a model to be considered as the best.
        - weights (Tuple[float, float]): The weights for the composite metric calculation Eg: (f1_weight, auroc_weight). Only used if best_metric_name is "composite". Default is (0.7, 0.3).

    Returns:
        - is_best (bool): True if the current model is the best model based on the specified metric and recall threshold, False otherwise.
        - updated_best_metric_value (float): The updated best metric value after comparing with the current model's metric value.

    Raises:
        - ValueError: If the best_metric_name is not one of the allowed options.
    """

    # validate metric name
    validate_metric_name(best_metric_name)

    # validate that current_metric_value is a float for single metrics and a tuple of (f1, auroc) for composite metric
    if best_metric_name == "composite":
        if (
            not isinstance(current_metric_value, tuple)
            or len(current_metric_value) != 2
        ):
            raise ValueError(
                "For 'composite' metric, current_metric_value must be a tuple of (f1, auroc)"
            )
    else:
        if not isinstance(current_metric_value, (int, float)):
            raise ValueError(
                f"For '{best_metric_name}' metric, current_metric_value must be a float"
            )

    if best_metric_name == "composite":
        # for composite metric calculate a weighted score of given metrics using the given weights (Eg: 0.7 f1 + 0.3 auroc) and compare to determine best model
        metric_1, metric_2 = current_metric_value
        score = weights[0] * metric_1 + weights[1] * metric_2
    else:
        # for single metric, directly compare the metric value to determine best model
        score = current_metric_value

    is_best = score > best_metric_value and recall >= recall_threshold

    return {
        "is_best": is_best,
        "updated_best_metric_value": score if is_best else best_metric_value,
        "current_metric_value": score,
    }


def validate_metric_name(metric_name: str) -> None:
    """
    Validates that the provided metric name is one of the allowed options.

    Args:
        - metric_name (str): The name of the metric to validate.
    Raises:
        - ValueError: If the metric name is not one of the allowed options.
    """
    allowed_metrics = [
        "composite",
        "auroc",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "loss",
    ]
    if metric_name not in allowed_metrics:
        raise ValueError(
            f"Invalid metric name '{metric_name}'. Allowed options are: {allowed_metrics}"
        )


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
    optimizer_name: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
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
        "optimizer_name": optimizer_name,
    }
    if extra is not None:
        config.update(extra)
    return config


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


def download_wandb_artifact(
    artifact_name: str,
    artifact_type: Literal["model", "dataset"],
    version: Optional[str] = "latest",
    local_download_path: Optional[str] = None,
) -> str:
    """
    Downloads a W&B artifact by name and saves it as a flat file (no subdirectory) in the specified local directory, then returns the full path to that file.

    Args:
        - artifact_name: Name of the W&B artifact (e.g. "baseline_model")
        - artifact_type: W&B artifact type. Must be "model" or "dataset"
        - version: Artifact version alias. Defaults to "latest"
        - local_download_path: Local path where the artifact should be downloaded. If None, uses the default path from config.

    Returns:
        - str: Full local path to the downloaded file e.g. "artifacts/models/baseline_model.pth"

    Raises:
        - ValueError        : If artifact_type is not "model" or "dataset"
        - FileNotFoundError : If the artifact is not found in W&B
        - FileNotFoundError : If the downloaded artifact directory contains no files
    """

    if artifact_type not in VALID_ARTIFACT_TYPES:
        raise ValueError(
            f"'artifact_type' must be one of {VALID_ARTIFACT_TYPES}. Got {artifact_type!r}."
        )

    cfg = load_config()

    # local path setup
    local_dir = (
        Path(local_download_path)
        if local_download_path
        else Path(cfg["paths"]["artifacts_dir"])
        / ("models" if artifact_type == "model" else "results")
    )

    # create local directory if it doesn't exist
    local_dir.mkdir(parents=True, exist_ok=True)

    project_name = cfg["wandb"]["project_name"]
    entry_name = cfg["wandb"]["entity"]

    # Check if already exists locally to prevent unnecessary download and W&B API calls
    existing = list(local_dir.glob(f"{artifact_name}.*"))
    if existing:
        print(
            f"[INFO] Artifact '{artifact_name}' already exists locally at "
            f"'{existing[0]}'. Skipping download...."
        )
        return str(existing[0])

    # If not found locally, download from W&B using the API (no run created) into a temp directory
    temp_dir = local_dir / f"_temp_{artifact_name}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    try:
        artifact = api.artifact(
            f"{entry_name}/{project_name}/{artifact_name}:{version}", type=artifact_type
        )

        artifact.download(root=str(temp_dir))

    except wandb.errors.CommError as exc:
        # Clean up empty staging dir before raising
        try:
            temp_dir.rmdir()
        except OSError:
            pass
        raise FileNotFoundError(
            f"Failed to download artifact '{artifact_name}:{version}' "
            f"from W&B project '{project_name}'.\n"
            f"W&B error: {exc}"
        ) from exc

    # Find the downloaded file
    # artifact.download() places files flat inside staging_dir
    downloaded_files = list(temp_dir.iterdir())
    if not downloaded_files:
        raise FileNotFoundError(
            f"Downloaded artifact '{artifact_name}' is empty. "
            f"Nothing found in staging directory '{temp_dir}'."
        )

    # Take the first file (each artifact is configured to hold exactly one file)
    downloaded_file = downloaded_files[0]
    extension = downloaded_file.suffix  # e.g. ".pth" or ".csv"

    # Move to flat canonical path: local_dir/<artifact_name>.<ext>
    final_path = local_dir / f"{artifact_name}{extension}"
    shutil.move(str(downloaded_file), str(final_path))

    try:
        temp_dir.rmdir()
    except OSError:
        pass  # not empty — leave

    print(f"[INFO] Artifact '{artifact_name}' downloaded to: '{final_path}'")
    return str(final_path)


def save_df_to_csv(
    df: pd.DataFrame,
    name: str,
    output_dir: Union[str, Path],
) -> str:
    """
    Saves a single DataFrame to a CSV file in the specified directory.
    Args:
        - df         : The DataFrame to save.
        - name       : Base file name with or without extension (e.g. "train_metrics")
        - output_dir : Directory in which to write the CSV. Created if absent.

    Returns:
        - str: The file path to the saved CSV.

    Raises:
        - TypeError : If df is not a pandas DataFrame.
        - ValueError: If name is an empty string.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("'name' must be a non-empty string.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / f"{Path(name).stem}.csv"
    df.to_csv(csv_path, index=False)

    print(f"[INFO] Saved '{name}' at {csv_path}")

    return str(csv_path)


def log_dfs_as_wandb_tables(
    dfs: List[pd.DataFrame],
    names: List[str],
    wandb_project: str,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
    active_run: Optional[wandb.sdk.wandb_run.Run] = None,
) -> None:
    """
    Logs multiple DataFrames as W&B Tables in a single wandb.log() call, ensuring all tables appear at the same step in the W&B UI.

    Args:
        - dfs            : List of DataFrames to log as W&B Tables.
        - names          : List of table names, one per DataFrame.
        - wandb_project  : W&B project name to log tables into.
        - wandb_run_name : Display name for the W&B run. Defaults to the first name in 'names' when not provided.
        - wandb_tags     : Optional list of tags to attach to the W&B run.
        - active_run     : Pass an already-initialised W&B run to reuse it instead of creating a new one.

    Raises:
        - ValueError: If dfs and names lists have different lengths, or if wandb_project is empty.
        - TypeError : If any item in dfs is not a DataFrame.

    Example (standalone):
        log_dfs_as_wandb_tables(
            dfs=[df_train, df_val],
            names=["train_metrics", "val_metrics"],
            wandb_project="my-project",
            wandb_run_name="table-log-run",
            wandb_tags=["tables"],
        )
    """

    if not wandb_project or not wandb_project.strip():
        raise ValueError("'wandb_project' must be a non-empty string.")
    if len(dfs) != len(names):
        raise ValueError(
            f"Number of DataFrames ({len(dfs)}) must match "
            f"number of names ({len(names)})."
        )
    for i, df in enumerate(dfs):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Item at index {i} in 'dfs' is not a DataFrame. "
                f"Got {type(df).__name__}."
            )

    run_name = wandb_run_name or names[0]
    owns_run = active_run is None

    if owns_run:
        wandb.init(
            project=wandb_project,
            name=run_name,
            tags=wandb_tags,
            job_type="table-logging",
        )

    try:
        # Batch all tables into a single wandb.log() — same step in the UI
        tables = {name: wandb.Table(dataframe=df) for df, name in zip(dfs, names)}
        wandb.log(tables)
        print(f"[INFO] Logged {len(tables)} table(s) to W&B: {list(tables.keys())}")
    finally:
        if owns_run:
            try:
                wandb.finish()
            except Exception:
                pass


def upload_artifacts_to_wandb(
    file_paths: List[str],
    names: List[str],
    wandb_project: str,
    artifact_type: str = "dataset",
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
    artifact_description: Optional[str] = None,
    active_run: Optional[wandb.sdk.wandb_run.Run] = None,
) -> None:
    """
    Uploads one or more files as independently versioned W&B Artifacts under a single W&B run.

    Each file becomes its own artifact identified by its name, giving each file an independent version history. Only the artifact whose content changes will receive a new version on subsequent uploads.

    Args:
        - file_paths           : List of local file paths to the files to upload.
        - names                : List of artifact names, one per file. Each name becomes the artifact's retrieval key (entity/project/name:alias).
        - wandb_project        : W&B project name to upload artifacts into.
        - artifact_type        : Type of artifact. Must be "dataset" or "model". Defaults to "dataset".
        - wandb_run_name       : Display name for the W&B run. Defaults to the first name in 'names' when not provided.
        - wandb_tags           : Optional list of tags to attach to the W&B run.
        - artifact_description : Optional human-readable description attached to every artifact uploaded in this call.
        - active_run           : Pass an already-initialised W&B run to reuse it. Not intended for direct use by callers.

    Raises:
        - ValueError: If artifact_type is not "dataset" or "model", list lengths mismatch, or wandb_project is empty.
        - ValueError : If any file path is not a string or if any file does not exist at the specified path.
        - ValueError: If any name is not a non-empty string, or if there are duplicate names (artifact names must be unique within a project).
        - ValueError: If any file path has no extension, as the filename (including extension) is used as the artifact's content key.

    Example (standalone):
        _upload_artifacts_to_wandb(
            csv_paths=["outputs/train_metrics.csv"],
            names=["train_metrics"],
            wandb_project="my-project",
            artifact_type="dataset",
            dfs_metadata=[df_train],
            wandb_run_name="artifact-upload-run",
            artifact_description="Training metrics export.",
        )

    Retrieval:
        api = wandb.Api()
        artifact = api.artifact("my-project/train_metrics:latest")
        artifact_dir = artifact.download()
        df = pd.read_csv(f"{artifact_dir}/train_metrics.csv")
    """

    # Validate artifact type
    if artifact_type not in VALID_ARTIFACT_TYPES:
        raise ValueError(
            f"'artifact_type' must be one of {VALID_ARTIFACT_TYPES}. "
            f"Got {artifact_type!r}."
        )

    # Validate W&B project name: must be non-empty when uploading artifacts
    if not wandb_project or not wandb_project.strip():
        raise ValueError("'wandb_project' must be a non-empty string.")

    # Validate list lengths to ensure each file has a corresponding name
    if len(file_paths) != len(names):
        raise ValueError(
            f"Number of file_paths ({len(file_paths)}) must match "
            f"number of names ({len(names)})."
        )

    # Validate that names are non-empty strings and unique (artifact names must be unique within a project)
    if len(names) != len(set(names)):
        duplicates = [n for n in names if names.count(n) > 1]
        raise ValueError(
            f"'names' contains duplicate values: {list(set(duplicates))}. "
            f"Each artifact must have a unique name."
        )

    # Validate that each file path has an extension
    for file_path in file_paths:
        if not Path(file_path).suffix:
            raise ValueError(
                f"File path '{file_path}' has no extension. "
                f"Please provide the full filename including extension "
                f"(e.g. 'outputs/model.pth', 'outputs/train_metrics.csv')."
            )

    run_name = wandb_run_name or names[0]
    owns_run = active_run is None

    if owns_run:
        wandb.init(
            project=wandb_project,
            name=run_name,
            tags=wandb_tags,
            job_type="artifact-upload",
        )

    try:
        for file_path, name in zip(file_paths, names):
            if not Path(file_path).exists():
                warnings.warn(
                    f"[WARNING] File '{file_path}' does not exist. "
                    f"Skipping upload to W&B for artifact '{name}'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            actual_filename = Path(file_path).name

            artifact = wandb.Artifact(
                name=name,
                type=artifact_type,
                description=artifact_description or None,
            )

            artifact.add_file(local_path=file_path, name=actual_filename)
            wandb.log_artifact(artifact)

            print(
                f"[INFO] Artifact '{name}' ({artifact_type}) uploaded "
                f"to W&B project '{wandb_project}'."
            )
    finally:
        if owns_run:
            try:
                wandb.finish()
            except Exception:
                pass


def save_and_track_dataframes(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    file_names: Union[str, List[str]],
    output_dir: Union[str, Path],
    save_as_artifact: bool = True,
    artifact_type: str = "dataset",
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
    artifact_description: Optional[str] = None,
) -> Union[str, List[str]]:
    """
    Saves one or more DataFrames as local CSV files and, when save_as_artifact=True, logs them to W&B as Tables and uploads each as an independently versioned Artifact — all under a single W&B run.

    When save_as_artifact=False, no W&B run is created at all — only local CSV files are written to disk.

    Args:
        - dfs                  : A single DataFrame or a list of DataFrames.
        - file_names           : A single name or a list of names (with or without '.csv' extension), one per DataFrame. Each name becomes both the local CSV filename
                                 and  the W&B artifact name.
        - output_dir           : Local directory where CSV files are saved.
        - save_as_artifact     : If True, logs W&B Tables and uploads each CSV as a separate W&B artifact under one run. If False, only local CSVs are written and no
                                 W&B run is created. Defaults to True.
        - artifact_type        : "dataset" or "model". Defaults to "dataset".
        - wandb_run_name       : Display name for the W&B run. Defaults to the first file name when not provided.
        - wandb_tags           : Optional list of tags for the W&B run.
        - artifact_description : Optional description attached to every artifact.

    Returns:
        - str or List[str]: A single path string for a single DataFrame input, or a list of path strings for multiple DataFrame inputs.

    Raises:
        - TypeError : If inputs contain unexpected types.
        - ValueError: If list lengths mismatch, names are empty, artifact_type is invalid, or wandb_project is missing when save_as_artifact=True.

    Example:
        paths = save_and_track_dataframes(
            dfs=[df_train, df_val],
            file_names=["train_metrics", "val_metrics"],
            output_dir="outputs/results",
            save_as_artifact=True,
            artifact_type="dataset",
            wandb_project="my-project",
            wandb_run_name="data-export",
            wandb_tags=["export", "metrics"],
        )
    """

    # Input normalization: allow single DataFrame and single name, but convert to lists for uniform processing
    if isinstance(dfs, pd.DataFrame):
        dfs_list = [dfs]
    elif isinstance(dfs, list):
        dfs_list = dfs
    else:
        raise TypeError(
            f"'dfs' must be a DataFrame or list of DataFrames. "
            f"Got {type(dfs).__name__}."
        )

    if isinstance(file_names, str):
        names_list = [file_names]
    elif isinstance(file_names, list):
        names_list = file_names
    else:
        raise TypeError(
            f"'file_names' must be a str or list of str. "
            f"Got {type(file_names).__name__}."
        )

    # Validate list lengths and types
    if len(dfs_list) != len(names_list):
        raise ValueError(
            f"Number of DataFrames ({len(dfs_list)}) must match "
            f"number of file names ({len(names_list)})."
        )
    for i, item in enumerate(dfs_list):
        if not isinstance(item, pd.DataFrame):
            raise TypeError(
                f"Item at index {i} in 'dfs' is not a DataFrame. "
                f"Got {type(item).__name__}."
            )
    cleaned_names: List[str] = []
    for i, n in enumerate(names_list):
        if not isinstance(n, str) or not n.strip():
            raise ValueError(f"file_names[{i}] is empty or invalid. Got: {n!r}")
        cleaned_names.append(Path(n).stem)

    # validate artifact_type and W&B project requirements
    if artifact_type not in VALID_ARTIFACT_TYPES:
        raise ValueError(
            f"'artifact_type' must be one of {VALID_ARTIFACT_TYPES}. "
            f"Got {artifact_type!r}."
        )
    if save_as_artifact and not wandb_project:
        raise ValueError("'wandb_project' must be provided when save_as_artifact=True.")

    # Saving locally only no W&B involvement
    if not save_as_artifact:
        csv_paths = [
            save_df_to_csv(df, name, output_dir)
            for df, name in zip(dfs_list, cleaned_names)
        ]
        print("[INFO] File saved locally without W&B tracking.")
        return csv_paths[0] if len(csv_paths) == 1 else csv_paths

    # Local save + W&B tables + W&B artifacts (single run)
    run_name = wandb_run_name or cleaned_names[0]

    wandb.init(
        project=wandb_project,
        name=run_name,
        tags=wandb_tags,
        job_type="data-upload",
    )

    try:
        # Step A — Save all CSVs locally first
        csv_paths = [
            save_df_to_csv(df, name, output_dir)
            for df, name in zip(dfs_list, cleaned_names)
        ]

        # Step B — Log all tables in one batched call, reusing the active run
        log_dfs_as_wandb_tables(
            dfs=dfs_list,
            names=cleaned_names,
            wandb_project=wandb_project,
            wandb_run_name=run_name,
            wandb_tags=wandb_tags,
            active_run=wandb.run,
        )

        # Step C — Upload each CSV as its own artifact, reusing the active run
        upload_artifacts_to_wandb(
            file_paths=csv_paths,
            names=cleaned_names,
            wandb_project=wandb_project,
            artifact_type=artifact_type,
            wandb_run_name=run_name,
            wandb_tags=wandb_tags,
            artifact_description=artifact_description,
            active_run=wandb.run,
        )

    finally:
        try:
            wandb.finish()
        except Exception:
            pass

    return csv_paths[0] if len(csv_paths) == 1 else csv_paths


def convert_model_training_results_to_df(
    training_config: Dict[str, Any],
    exclude_keys: Optional[List[str]] = None,
    save_df_as_csv: bool = True,
    csv_file_name: Optional[Union[str, List[str]]] = [
        "best_model_results",
        "all_model_results",
    ],
    wandb_run_name: Optional[str] = "model_results_summary",
    wandb_tags: Optional[List[str]] = None,
    save_as_wandb_artifact: bool = True,
    artifact_description: Optional[str] = None,
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
        best_model_metric_value = 0.0
        for epoch, eval_res in enumerate(
            results["eval"], start=1
        ):  # start counting from 1
            current_model_metric_value = (
                (
                    eval_res.get("f1_score", 0),
                    eval_res.get("auroc", 0),
                )
                if base_model_info["best_model_metric_name"] == "composite"
                else eval_res.get(base_model_info["best_model_metric_name"], 0)
            )
            model_checkpoints = is_best_model(
                best_metric_name=base_model_info["best_model_metric_name"],
                current_metric_value=current_model_metric_value,
                best_metric_value=best_model_metric_value,
                recall=eval_res.get("recall", 0),
                recall_threshold=recall_threshold,
            )

            best_model_metric_value = model_checkpoints["updated_best_metric_value"]

            if model_checkpoints["is_best"]:
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
        results_artifacts_dir = Path(cfg["paths"]["artifacts_dir"]) / "results"
        project_name = cfg["wandb"]["project_name"]

        saved_paths = save_and_track_dataframes(
            dfs=[best_model_df, all_model_df],
            file_names=csv_file_name,
            output_dir=results_artifacts_dir,
            save_as_artifact=save_as_wandb_artifact,
            artifact_type="dataset",
            wandb_project=project_name,
            wandb_run_name=wandb_run_name,
            wandb_tags=wandb_tags if wandb_tags else [],
            artifact_description=artifact_description,
        )

    return {
        "best_model_df": best_model_df,
        "all_model_df": all_model_df,
        "local_saved_paths": saved_paths,
    }
