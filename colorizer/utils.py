import os
import random
from collections import namedtuple
from typing import NamedTuple

import numpy as np
import torch
from kornia.color.lab import rgb_to_lab
from PIL import Image
from torchvision import io
from torchvision.transforms import v2
from torchvision.utils import make_grid


def rgb_to_zero_centered_normalized_lab(
    images: torch.Tensor, split_light_and_color: bool = True
) -> torch.Tensor | tuple[torch.Tensor]:
    """Transforms an RGB image to LAB color space and normalizes
    the LAB channels between -1 to 1

    Parameters
    ----------
    images : torch.Tensor
        Batch of images in RGB Format with shape (b, c, h, w)

    split_light_and_color : bool, default=True
        If True returns a tuple of the light channel and AB channels
        concatenated, else returns a single LAB image

    Returns
    -------
    Normalized lab or tuple of light and ab channel tensors
    """
    lab_images = rgb_to_lab(images)
    light_ch = (lab_images[:, [0], :, :] / 50) - 1
    ab_ch = lab_images[:, [1, 2], :, :] / 110

    if not split_light_and_color:
        return torch.concat([light_ch, ab_ch], dim=1)
    return light_ch, ab_ch


def get_absolute_path_given_relative(relative_path: str) -> str:
    """Given a relative path, creates an absolute path from it

    Parameters
    ----------
    relative_path : str
        relative file from the current working directory

    Returns
    -------
    Absolute path based on the relative path
    """
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
    """Loads a list of validation images to tensors

    Parameters
    ----------
    images_paths : list[str]
        List of images paths to be loaded into tensors

    resize : int
        Resize all the images to a specific squared shape

    load_as_rgb : bool, default=False
        Images can be loaded as grayscale (single channel) or as RGB (three channels)

    max_images : int, default=16
        Load only `max_images` of the total images in `images_paths`

    random_sample : bool, default=False
        If False, the first images in the list up to `max_images` will be loaded. Else
        a random sample of ordered images will be used instead

    seed : int, default=None
        When `random_sample` is True, use a seed for deterministic samples

    Returns
    -------
    batch of torch.Tensor images
    """
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


def evaluate_model(
    pred_images: torch.tensor, step: int, nrow: int = 4, save_dir: str | None = None
) -> np.ndarray:
    """Creates a grid with the predicted images for easier visualization
    and visual evaluation of the model training process and results

    Parameters
    ----------
    pred_images : torch.Tensor
        Images predicted by the model in tensor format

    step : int
        Step of the training process. Will only influence the name of the saved
        image if `save_dir` is enabled

    nrow : int, default=4
        Maximum amount of rows in the grid, additional images will be placed in
        columns

    save_dir : str, default=None
        If valid, saves the grid with results to a local file at this directory

    Returns
    -------
    images grid of type numpy.ndarray
    - single image composed of multiple smaller images
    """
    grid = make_grid(pred_images, nrow=nrow, padding=1).permute(1, 2, 0).numpy()
    if save_dir is not None:
        image_grid = Image.fromarray((grid.numpy() * 255).astype(np.uint8))
        os.makedirs(save_dir, exist_ok=True)
        image_grid.save(os.path.join(save_dir, f"step-{step}.png"))
    return grid.numpy()


def prepare_image_for_inference(
    path: str, input_size: int
) -> NamedTuple("Output", ("image", torch.Tensor), ("original_shape", tuple[int, int])):
    """Given an image file path, preprocess the image for inference.

    Parameters
    ----------
    path : str
        path of the image to be used in inference

    input_size : int
        Input size required by the inference pipeline

    Returns
    -------
    NamedTuple composed of:
    image: preprocessed tensor representation of the input image
    original_shape: original height and width of the input image
    """
    img_tensor = io.read_image(path=path, mode=io.ImageReadMode.GRAY) / 255
    _, original_height, original_width = img_tensor.shape
    original_shape = (original_height, original_width)
    img_tensor = (img_tensor - 0.5) * 2
    img_tensor = v2.Resize(size=(input_size, input_size), antialias=True)(img_tensor)
    output = namedtuple("Output", ["image", "original_shape"])
    return output(img_tensor, original_shape)
