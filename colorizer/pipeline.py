import torch
from diffusers import DDPMScheduler, DiffusionPipeline, ImagePipelineOutput, UNet2DModel
from diffusers.utils.torch_utils import randn_tensor
from kornia.color.lab import lab_to_rgb


class Pix2PixColorizerPipeline(DiffusionPipeline):
    """Custom diffusion pipeline for colorizing grayscale images

    Parameters
    ----------
    unet : UNet2DModel
        Unconditioned UNet model used for predicting noise

    scheduler : DDPMScheduler
        Noise scheduler used to calculate previous samples based on
        unet's predictions
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet: UNet2DModel, scheduler: DDPMScheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        bw_images: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None = None,
        num_inference_steps: int = 1000,
        return_tuple: bool = False,
    ) -> ImagePipelineOutput | tuple[torch.Tensor]:
        """Prediction pipeline given grayscale images

        Parameters
        ----------
        bw_images : torch.tensor
            Input grayscale images with shape (b, c, h, w)

        generator : torch.Generator
            Generator used for settings a seed and making noise generation
            deterministic

        num_inference_steps : int, default=1000
            Total denoising steps

        return_tuple : bool, default=False
            If set the pipeline returns a tuple with predicted images instead
            of ImagePipelineOutput class with the predictions
        """
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

        output = torch.concat([lightness, ab], dim=1)
        output = lab_to_rgb(output)
        output = output.cpu()

        if return_tuple:
            return (output,)

        return ImagePipelineOutput(images=output)
