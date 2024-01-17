import torch
from diffusers import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from kornia.color.lab import lab_to_rgb


class Pix2PixColorizerPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        bw_images: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None = None,
        num_inference_steps: int = 2000,
        return_dict: bool = True,
    ) -> ImagePipelineOutput | tuple:

        bw_images = bw_images.to(self.device)  # range -1 to 1
        # Sample gaussian noise to begin loop
        batch_size = bw_images.shape[0]
        channels = 2  # a & b channels

        image_shape = (
            batch_size,
            channels,
            self.unet.config.sample_size,
            self.unet.config.sample_size,
        )

        ab = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            # Model outputs the ab dimension
            # we need to concatenate the rest of the image

            previous_ab = ab
            # (b, 3, h, w) | -1 to 1
            images = torch.concat([bw_images, previous_ab], dim=1)

            # (b, 2, h, w)
            predicted_ab = self.unet(images, t).sample

            # 2. compute previous image: x_t -> x_t-1
            # (b, 2, h, w)
            ab = self.scheduler.step(
                predicted_ab, t, previous_ab, generator=generator
            ).prev_sample

        lightness = (bw_images + 1) * 50
        ab = ab * 110

        images = torch.concat([lightness, ab], dim=1)
        images = lab_to_rgb(images)
        images = images.cpu()

        if not return_dict:
            return (images,)

        return ImagePipelineOutput(images=images)
