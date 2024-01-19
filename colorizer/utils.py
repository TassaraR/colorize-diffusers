import torch
from kornia.color.lab import rgb_to_lab


def rgb_to_zero_centered_normalized_lab(
    images: torch.Tensor, split_light_and_color: bool = True
) -> torch.Tensor | tuple[torch.Tensor]:

    lab_images = rgb_to_lab(images)
    light_ch = (lab_images[:, [0], :, :] / 50) - 1
    ab_ch = lab_images[:, [1, 2], :, :] / 110

    if not split_light_and_color:
        return torch.concat([light_ch, ab_ch], dim=1)
    return light_ch, ab_ch
