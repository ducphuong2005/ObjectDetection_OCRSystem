
import albumentations as A
import cv2
import numpy as np
from typing import List, Tuple


def get_train_augmentations() -> A.Compose:
    """Pipeline augmentation cho tập train: xoay, blur, thay đổi sáng/tương phản."""
    return A.Compose([
        A.HorizontalFlip(p=0.3),
        A.RandomRotate90(p=0.2),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT,
                 value=(255, 255, 255), p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=1.0
            ),
            A.CLAHE(clip_limit=2.0, p=1.0),
        ], p=0.4),
        A.ToGray(p=0.2),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['category_ids'],
        min_area=100,
        min_visibility=0.3,
    ))


def get_val_augmentations() -> A.Compose:
    """Pipeline cho tập val — không augment, chỉ pass qua."""
    return A.Compose([], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['category_ids'],
    ))


def apply_augmentation(
    image: np.ndarray,
    bboxes: List[List[float]],
    category_ids: List[int],
    is_train: bool = True
) -> Tuple[np.ndarray, List[List[float]], List[int]]:
    """Áp dụng augmentation lên ảnh và bounding boxes."""
    transform = get_train_augmentations() if is_train else get_val_augmentations()

    transformed = transform(
        image=image,
        bboxes=bboxes,
        category_ids=category_ids,
    )

    return (
        transformed['image'],
        transformed['bboxes'],
        transformed['category_ids'],
    )
