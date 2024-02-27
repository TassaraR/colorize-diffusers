# Colorize Diffuser

Colorizer model using Pix2Pix diffusion based on Hugging Face's Diffusers framework

## How does it work?

Based on a grayscale image we try to predict its colors.


Usually diffusion models are composed of multiple components and models. This one requires two of them:
- **UNet**:
  - Predicts the noise $n$ at each timestep $t$


- **Noise Scheduler**:
    - During training it adds $n$ amount of noise at each timestep $t$, gradually transforming the image into pure noise
    - During inference it guides the denoising process, which aims to recover the clean image from the noisy input.


In this case in order to condition model we will not try to generate an image based on pure noise but rather from the concatenation of pure noise (the channels we are looking to predict <color>) and the grayscale image.

The main goal is to manage to denoise the input data in order to produce the colors from the noisy channels.

For this model we work in the $LAB$/$CIELAB$ colorspace. Where one channel corresponds to lightness `range: [0, 100]` and the ab channels to the "color" of the image with an approximate range or `[-128, 127]`.

Its not necessary to use the $LAB$ colorspace in this task but it does simplify the model a lot. With RGB we would have an input size between 4-6 (3 noisy channels to predict RGB, and either 1 or 3 channels for the concatenated base grayscale image), but with $LAB$ we only need to predict two channels (a & b) and the lightness channel works the same as a grayscale image, we just need to rescale it.
