"""
Contains:
    - function to download the data from kaggle
    - ChestXRayDataset class
"""

import shutil
from pathlib import Path

import kagglehub


def download_data(target_dir: str, platform: str) -> Path:
    """
    Prepares the chest X-ray dataset at target_dir with train/ and test/ splits,
    merging the original val/ images into train/.

    On Colab  : downloads from Kaggle via kagglehub, then copies and organizes.
    On Kaggle : dataset is pre-mounted at /kaggle/input/ — copies and organizes
                directly from there (no download needed).

    The copy + val-merge logic is identical for both platforms.
    Only the source path differs.

    Args:
        target_dir : path where the organized dataset will be placed
        platform   : "colab" or "kaggle"

    Returns:
        Path : path to the organized chest_xray/ directory containing
               train/ and test/ splits. Returns None on failure.
    """
    target_dir = Path(target_dir)
    final_data_path = target_dir / "chest_xray"

    # ── 1. Early exit: data already organized
    if (final_data_path / "train").exists():
        print(
            f"[INFO] Dataset already exists at {final_data_path}. Skipping.Download.."
        )
        return final_data_path

    # 2. Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 3. Resolve source path (platform-dependent)
        if platform == "colab":
            # kagglehub downloads into a global cache and returns its path
            print("[INFO] Downloading dataset from Kaggle via kagglehub...")
            cache_path = Path(
                kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
            )
            # The zip has a nested structure: chest_xray/chest_xray/[train, test, val]
            source_content_path = cache_path / "chest_xray" / "chest_xray"
            print("[INFO] Download complete.")

        elif platform == "kaggle":
            # Dataset is pre-mounted — no download needed
            # Structure: /kaggle/input/chest-xray-pneumonia/chest_xray/[train, test, val]
            source_content_path = Path("/kaggle/input/chest-xray-pneumonia/chest_xray")
            print(f"[INFO] Using pre-mounted dataset at {source_content_path}")

        # 4. Validate source exists before proceeding
        if not source_content_path.exists():
            raise FileNotFoundError(
                f"[ERROR] Source data not found at {source_content_path}. "
                f"On Kaggle, ensure the dataset 'paultimothymooney/chest-xray-pneumonia' "
                f"is added as input to this notebook."
            )

        # 5. Copy train/ and test/ from source to target_dir
        # (identical for both platforms from this point onward)
        print(f"[INFO] Organizing data into {final_data_path}...")
        final_data_path.mkdir(parents=True, exist_ok=True)

        for split in ["train", "test"]:
            src = source_content_path / split
            dst = final_data_path / split

            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                print(f"[INFO] Copied '{split}' split successfully.")

        # 6. Merge val/ into train/
        val_src = source_content_path / "val"
        if val_src.exists():
            for class_dir in val_src.iterdir():  # NORMAL, PNEUMONIA
                if class_dir.is_dir():
                    train_class_dst = final_data_path / "train" / class_dir.name
                    train_class_dst.mkdir(parents=True, exist_ok=True)

                    for img_file in class_dir.iterdir():
                        dst_file = train_class_dst / img_file.name
                        # Rename on collision to avoid silent overwrites
                        if dst_file.exists():
                            dst_file = train_class_dst / f"val_{img_file.name}"
                        shutil.copy2(img_file, dst_file)

            print("[INFO] Merged val/ into train/ successfully.")

        print(f"[SUCCESS] Dataset ready at: {final_data_path}")

    except Exception as e:
        print(f"[ERROR] {e}")
        return None

    return final_data_path


from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from .transform import basic_transform


class ChestXRayDataset(torch.utils.data.Dataset):
    """
    A Dataset wrapper that applies Albumentations transforms to an
    ImageFolder or Subset instance.

    NOTE:
        - If no transfom is given then applies ToTensorV2() transform and convert to tensor. But doesn't normalize or scale.
        - The base ImageFolder must be created with transform=None to avoid double-transform issues.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        transform: Callable[[Any], Any] = basic_transform,
    ) -> None:

        self.dataset = dataset
        self.transform = transform

        if isinstance(self.dataset, torch.utils.data.Subset):
            # dataset is an instance of Subset class
            self.samples = [
                self.dataset.dataset.samples[i] for i in self.dataset.indices
            ]
            self.class_to_idx = self.dataset.dataset.class_to_idx
            self.classes = self.dataset.dataset.classes

        else:
            # dataset is an instance of Dataset class
            self.samples = dataset.samples
            self.classes = dataset.classes
            self.class_to_idx = dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        # fetch image as PIL image and convert to np array (H, W, C)
        image = np.array(Image.open(img_path).convert("RGB"))

        # apply transform
        image = self.transform(image=image)["image"]  # return Tensor obj (C, H, W)

        return image, label
