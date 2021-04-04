# From https://www.kaggle.com/haqishen/augmix-based-on-albumentations
from typing import Optional

import albumentations as A
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageEnhance, ImageOps

from .img_augs import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.0


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR
    )


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR
    )


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR
    )


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR
    )


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


AUGMENTATIONS = [
    autocontrast,
    equalize,
    posterize,
    rotate,
    solarize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
]

AUGMENTATIONS_ALL = [
    autocontrast,
    equalize,
    posterize,
    rotate,
    solarize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
    color,
    contrast,
    brightness,
    sharpness,
]


def normalize(image):
    """Normalize input image channel-wise to zero mean and unit variance."""
    return image - 127


def apply_op(image, op, severity):
    #   image = np.clip(image, 0, 255)
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img)


def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.0):
    """Perform AugMix augmentations and compute mixture.
    Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
    mixed: Augmented and mixed image.
    """
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image).astype(np.float32)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(AUGMENTATIONS_ALL)
            image_aug = apply_op(image_aug, op, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * image_aug
    #         mix += ws[i] * normalize(image_aug)

    mixed = (1 - m) * image + m * mix
    #     mixed = (1 - m) * normalize(image) + m * mix
    return mixed


class RandomAugMix(ImageOnlyTransform):
    def __init__(
        self, severity=3, width=3, depth=-1, alpha=1.0, always_apply=False, p=0.5
    ):
        super().__init__(always_apply, p)
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def apply(self, image, **params):
        image = augment_and_mix(
            image, self.severity, self.width, self.depth, self.alpha
        )
        return image


class RandomAug(ImageOnlyTransform):
    def __init__(self, severity=3, n=3, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.severity = severity
        self.n = n
        self.augment_list = AUGMENTATIONS_ALL

    def apply(self, image, **ignored):
        ops = np.random.choice(self.augment_list, self.n)
        for op in ops:
            image = apply_op(image, op, self.severity)
        return image


def make_aug(
    img_size: int,
    crop_size: Optional[int],
    starategy: str,
    mode: str,
    severity: int = 3,
    width: int = 3,
):
    assert mode in ["train", "val", "test"]
    assert starategy in ["mix", "rand"]
    if crop_size is None:
        crop_size = img_size

    if starategy == "mix":
        rand_aug = RandomAugMix(severity=severity, width=width, alpha=1.0, p=1.0)
    else:
        rand_aug = RandomAug(severity=severity, n=width, p=0.75)

    if mode == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    crop_size,
                    crop_size,
                    scale=(0.9, 1),
                    p=1,
                    interpolation=cv2.INTER_LANCZOS4,
                ),
                rand_aug,
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
                A.Resize(img_size, img_size, interpolation=cv2.INTER_LANCZOS4),
                A.CenterCrop(crop_size, crop_size, p=1 if crop_size != img_size else 0),
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2(),
            ],
        )
