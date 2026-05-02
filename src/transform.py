
"""
This file contains transforms for train and test data
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224

# transform for test and validation set
test_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


# transform for training set
train_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),

    # geometrical variations
    A.HorizontalFlip(p=0.4),
    A.Rotate(limit=10, p=0.5),  # ±10% rotation
    A.Affine(
        translate_percent=0.05,       # ±5% shift
        scale=(0.95, 1.05),           # ±5% zoom
        rotate=0,                     # 0 rotation (done above)
        p=0.4
    ),

    # simulate scanner condition & quality
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.4
    ),

    # randomly remove rect. patches
    A.CoarseDropout(
        num_holes_range=(1, 8),
        hole_height_range=(8, 16),
        hole_width_range=(8, 16),
        fill=0,     # fill with black pixels
        p=0.3
    ),

    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  #(input - mean) / std
    ToTensorV2(),
])

# basic transform to convert to tensor object
basic_transform = A.Compose([
    ToTensorV2()
])
