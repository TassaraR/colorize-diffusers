import os
import random

import torch
from kornia.color.lab import rgb_to_lab
from torchvision import io
from torchvision.transforms import v2


def rgb_to_zero_centered_normalized_lab(
    images: torch.Tensor, split_light_and_color: bool = True
) -> torch.Tensor | tuple[torch.Tensor]:

    lab_images = rgb_to_lab(images)
    light_ch = (lab_images[:, [0], :, :] / 50) - 1
    ab_ch = lab_images[:, [1, 2], :, :] / 110

    if not split_light_and_color:
        return torch.concat([light_ch, ab_ch], dim=1)
    return light_ch, ab_ch


def get_absolute_path_given_relative(relative_path: str) -> str:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    abs_path = os.path.join(dir_path, relative_path)
    abs_path = os.path.normpath(abs_path)
    return abs_path


def load_validation_images(
    images_paths: list[str],
    resize: int,
    load_as_rgb: bool = False,
    max_images: int = 16,
    random_sample: bool = False,
    seed: int | None = None,
) -> torch.Tensor:

    mode = io.ImageReadMode.GRAY
    norm_mean = [0.5]
    norm_std = [0.5]
    if load_as_rgb:
        mode = io.ImageReadMode.RGB
        norm_mean *= 3
        norm_std *= 3

    transform = v2.Compose(
        [
            v2.Resize((resize, resize), antialias=True),
            v2.Lambda(lambda im: im / 255),
            v2.Normalize(mean=norm_mean, std=norm_std),
        ]
    )

    if random_sample:
        if seed is not None:
            random.seed(seed)
        images_paths = random.sample(images_paths, min(max_images, len(images_paths)))
        images_paths.sort()
    else:
        images_paths = images_paths[:max_images]

    images = [transform(io.read_image(path, mode=mode)) for path in images_paths]
    batch = torch.stack(images)
    return batch
