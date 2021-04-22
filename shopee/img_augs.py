from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_albu_augs(img_size: int, crop_size: Optional[int], mode: str):
    assert mode in ["train", "val", "test"]
    if crop_size is None:
        crop_size = img_size
    if mode == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(crop_size, crop_size, scale=(0.9, 1), p=1,),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.7),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.7
                ),
                A.CLAHE(clip_limit=(1, 4), p=0.5),
                A.OneOf(
                    [
                        A.OpticalDistortion(distort_limit=1.0),
                        A.GridDistortion(num_steps=5, distort_limit=1.0),
                        A.ElasticTransform(alpha=3),
                    ],
                    p=0.4,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=[10, 50]),
                        A.GaussianBlur(),
                        A.MotionBlur(),
                        A.MedianBlur(),
                    ],
                    p=0.4,
                ),
                A.ToGray(p=0.5),
                A.ToSepia(p=0.5),
                A.Cutout(
                    max_h_size=int(crop_size * 0.075),
                    max_w_size=int(crop_size * 0.075),
                    num_holes=10,
                    p=0.5,
                ),
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ],
        )
    if mode in ["val", "test"]:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.CenterCrop(crop_size, crop_size, p=1 if crop_size != img_size else 0),
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ],
        )
