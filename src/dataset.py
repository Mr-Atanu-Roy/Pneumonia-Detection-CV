"""
Contains:
    - function to download the data from kaggle
    - ChestXRayDataset class
"""

import shutil
import os
import kagglehub
from pathlib import Path

def download_data(target_dir: str) -> Path:
    """
    Download the data from Kaggle using kagglehub, moves the train, test data it to target_dir. Merge the val and train data and returns the path where train, test splits are placed.

    Args:
        - target_dir (str): local path where the data will be stored
    Returns:
        - Path: path where the train, test, val splits are placed
    """

    target_dir = Path(target_dir)

    # The final location where train, test, val splits will be placed
    final_data_path = target_dir / "chest_xray"

    #1. Check if the final organized data already exists
    if (final_data_path / "train").exists():
        print(f"[INFO] Dataset already exists at {final_data_path}. Skipping download...")
        return final_data_path

    # 2. Create target directory if it doesn't exist
    if not target_dir.exists():
        print(f"[INFO] Creating directory at {target_dir}...")
        target_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("[INFO] Starting download from Kaggle via kagglehub...")
        # kagglehub downloads and unzipping into a global cache
        cache_path = Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia"))

        # 3. Define the source of the nested mess in the cache
        # The structure in this specific zip is: chest_xray/chest_xray/[train, test, val]
        source_content_path = cache_path / "chest_xray" / "chest_xray"

        print(f"[INFO] Finished Downloading.")
        print(f"[INFO] Organizing and moving data to {final_data_path}...")

        # Create the final folder if it doesn't exist
        final_data_path.mkdir(parents=True, exist_ok=True)

        # 4. Move train, test folders from cache to target_dir. Merge train and val
        for split in ["train", "test"]:
            src = source_content_path / split
            dst = final_data_path / split

            if src.exists():
                # If the destination already exists: remove old data and move the new data
                if dst.exists():
                    shutil.rmtree(dst)

                # Copy from cache to dest
                shutil.copytree(src, dst)
                print(f"[INFO] Successfully moved {split} split.")

        # Merge val/NORMAL and val/PNEUMONIA into train/NORMAL and train/PNEUMONIA respectively
        val_src = source_content_path / "val"
        if val_src.exists():
            for class_dir in val_src.iterdir():          # NORMAL, PNEUMONIA
                if class_dir.is_dir():
                    train_class_dst = final_data_path / "train" / class_dir.name
                    train_class_dst.mkdir(parents=True, exist_ok=True)

                    for img_file in class_dir.iterdir():
                        dst_file = train_class_dst / img_file.name
                        # Rename on collision to avoid silent overwrites
                        if dst_file.exists():
                            dst_file = train_class_dst / f"val_{img_file.name}"

                        shutil.copy2(img_file, dst_file)

            print("[INFO] Successfully merged val and train splits.")


        print("[SUCCESS]  Dataset organized.")

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        return None

    print(f"[INFO] Data directory is ready at: {final_data_path}")
    return final_data_path

import torch
import numpy as np
from typing import Callable, Any
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

    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 transform:  Callable[[Any], Any]=basic_transform) -> None:

        self.dataset = dataset
        self.transform = transform

        if isinstance(self.dataset, torch.utils.data.Subset):
            # dataset is an instance of Subset class
            self.samples = [self.dataset.dataset.samples[i] for i in self.dataset.indices]
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
        image = self.transform(image=image)["image"]    #return Tensor obj (C, H, W)

        return image, label
