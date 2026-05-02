"""This file contains:
    - functions to create dataloaders
    - function to split train data into train and validation splits
    - function to calculate the weightage of pos and neg class in a dataloader
"""

import os
from typing import Callable, Tuple, List, Any, Optional

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from .dataset import ChestXRayDataset


def create_dataloaders(batch_size: int,
                       val_size: float,
                       num_workers: int,
                       persistent_workers: bool,
                       train_val_dir: Optional[str]=None,
                       test_dir: Optional[str]=None,
                       train_transform: Optional[Callable[[Any], Any]]=None,
                       test_transform: Optional[Callable[[Any], Any]]=None,
                       dataloader_type: str="all") -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:

    """
    1. Creates train and val splits subsets based on val_size
    2. Creates train, val and test datasets with given transforms
    3. Creates and returns train, val and test dataloaders with given batch size and num workers
    NOTE:
        - If dataloader_type="all" then train, val and test dataloaders are given
        - If dataloader_type="train" then only train and val dataloaders are given
        - If dataloader_type="test" then only test dataloader is given

    Args:
        - batch_size (int): batch size for the dataloaders
        - val_size (float): percentage of data to be used for validation set
        - num_workers (int): number of workers to be used for the dataloaders
        - persistent_workers (bool): whether to use persistent workers for the dataloaders
        - train_val_dir (str): path to the train and val data
        - test_dir (str): path to the test data
        - train_transform (Callable): transform to be applied to the train data
        - test_transform (Callable): transform to be applied to the test data
        - dataloader_type (str): type of dataloader to be created. Can be "all", "train" or "test"

    Returns:
        - train_dataloader, val_dataloader, test_dataloader (torch.utils.data.DataLoader): train, val and test dataloaders

    Raises:
        - ValueError: if dataloader_type is not in ["all", "train", "test"]
    """

    # validate dataloader_type
    if dataloader_type not in ["all", "train", "test"]:
        raise ValueError(
            f"dataloader_type must be in ['all', 'train', 'test']. ({dataloader_type} given)"
        )


    if dataloader_type in ["all", "train"]:

        # train_transform and train_val_dir is required
        if train_transform is None or train_val_dir is None:
            raise ValueError(
                f"train_transform and train_val_dir must be provided for {dataloader_type} dataloader_type"
            )

        #creating train and val splits and getting the train and val subsets
        train_subset, val_subset = create_test_val_split(
            train_val_dir=train_val_dir,
            val_size=val_size
        )

        #create the transformed dataset for train and val splits
        train_dataset = ChestXRayDataset(dataset=train_subset, transform=train_transform)
        val_dataset = ChestXRayDataset(dataset=val_subset, transform=test_transform)

        #creating train and val dataloaders
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True
        )

        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True
        )

        class_names = train_dataset.classes


        # return only train dataloader if dataloader_type is 'train'
        if dataloader_type == "train":
            return train_dataloader, val_dataloader, class_names


    if dataloader_type in ["all", "test"]:

        # test_transform and test_val_dir is required
        if test_transform is None or test_dir is None:
            raise ValueError(
                f"test_transform and test_dir must be provided for {dataloader_type} dataloader_type"
            )

        #create test dataset
        test_dataset = ChestXRayDataset(dataset=ImageFolder(test_dir, transform=None), transform=test_transform)

        #create test dataloader
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True
        )

        class_names = test_dataset.classes

        # return only test dataloader if dataloader_type is 'test'
        if dataloader_type == "test":
            return test_dataloader, class_names

    return train_dataloader, val_dataloader, test_dataloader, class_names



from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

def create_test_val_split(train_val_dir: str,
                          val_size: float) -> Tuple[Subset, Subset]:
    """
    Splits the data into training and validation split preserving the class balance
    Args:
        - train_val_dir (str): path to the train and val data
        - val_size (float): percentage of data to be used for validation

    Returns:
        - train_subset (Subset): train subset of the data
        - val_subset (Subset): validation subset of the data

    NOTE:
        - Here no transform is used as the train and val will require different transforms. Use the TransformedDataset class for creating a transfomed dataset.
    """

    # 1. Create the dataset for train and val data
    train_val_dataset = ImageFolder(root=train_val_dir, transform=None)

    # 2. Split into train and val splits
    targets = train_val_dataset.targets #get list of all targets

    train_idx, val_idx = train_test_split(
        range(len(train_val_dataset)),
        test_size=val_size,
        stratify=targets,   #ensure the class balance while spliting
        random_state=42
    )

    #get the train and val subsets
    train_subset = Subset(train_val_dataset, train_idx)
    val_subset = Subset(train_val_dataset, val_idx)

    return train_subset, val_subset


def get_class_weights(dataloader: DataLoader,
                      eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes pos and neg class weights from a dataloader in a single pass.

    Args:
        - dataloader (DataLoader): dataloader to compute weights from
        - eps (float): small value to avoid division by zero

    Returns:
        - tuple: (pos_weight, neg_weight) as float32 tensors
    """

    total_pos, total_neg = 0, 0

    for _, labels in dataloader:
        labels = labels.view(-1)
        total_pos += (labels == 1).sum().item()
        total_neg += (labels == 0).sum().item()

    total = total_pos + total_neg

    pos_weight = torch.tensor(total / (2 * total_pos + eps), dtype=torch.float32)
    neg_weight = torch.tensor(total / (2 * total_neg + eps), dtype=torch.float32)

    return pos_weight, neg_weight
