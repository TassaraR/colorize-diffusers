import math
import os
import shutil
from collections.abc import Callable

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.training_utils import EMAModel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm

import colorizer.utils as utils
from colorizer.inference import Pix2PixColorizerPipeline


class Trainer:
    def __init__(
        self,
        unet: UNet2DModel,
        noise_scheduler: DDPMScheduler,
        ema: EMAModel,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        epochs: int,
        gradient_accumulation_steps: int = 1,
        max_train_steps: int | None = None,
        lr_scheduler: Callable | None = None,
        checkpointing_steps: int | None = None,
        checkpoints_total_limit: int | None = None,
        checkpoints_output_dir: str | None = None,
        accelerator_kwargs: dict | None = None,
        tracker_experiment_name: str | None = None,
        tracker_log_directory: str = "logs",
    ):

        self.accelerator.init_trackers(tracker_experiment_name)
        default_accelerator_settings = {
            "mixed_precision": "fp16",
            "log_with": "tensorboard",
            "project_dir": tracker_log_directory,
        }
        acc_kwargs = accelerator_kwargs if accelerator_kwargs else {}
        _accelerator_kwargs = {**default_accelerator_settings, **acc_kwargs}
        self.accelerator = Accelerator(**_accelerator_kwargs)
        self.accelerator.gradient_accumulation_steps = gradient_accumulation_steps

        # Model related variables
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.train_dataloader = train_dataloader
        self.ema = ema
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        (self.unet, self.optimizer, self.train_dataloader,) = self.accelerator.prepare(
            self.unet,
            self.optimizer,
            self.train_dataloader,
        )
        if lr_scheduler:
            self.lr_scheduler = self.accelerator.prepare_scheduler(self.lr_scheduler)

        # Training steps
        self.epochs = epochs
        self.global_step = 0
        self.first_epoch = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_train_steps = max_train_steps
        self.overrode_max_train_steps = False
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.gradient_accumulation_steps
        )
        self._recompute_training_steps()

        # Checkpoint related variables
        self.checkpointing_steps = checkpointing_steps
        self.checkpoints_total_limit = checkpoints_total_limit
        self.resume_from_checkpoint = False
        self.checkpoints_output_dir = checkpoints_output_dir
        self.resume_step = -1

    def _recompute_training_steps(self) -> None:
        if self.max_train_steps is None:
            self.max_train_steps = self.epochs * self.num_update_steps_per_epoch
            self.overrode_max_train_steps = True

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.gradient_accumulation_steps
        )
        if self.overrode_max_train_steps:
            self.max_train_steps = self.epochs * self.num_update_steps_per_epoch
        self.epochs = math.ceil(self.max_train_steps / self.num_update_steps_per_epoch)

    def load_checkpoint(self, path: str) -> None:
        self.accelerator.load_state(path)
        self.global_step = int(path.split("-")[-1])
        resume_global_step = self.global_step * self.gradient_accumulation_steps
        self.first_epoch = self.global_step // self.num_update_steps_per_epoch
        self.resume_step = (
            resume_global_step
            % self.num_update_steps_per_epoch
            * self.gradient_accumulation_steps
        )
        self.resume_from_checkpoint = True

    def _skip_step(self, step: int) -> bool:
        skip_condition = (
            self.resume_from_checkpoint
            and self.epochs == self.first_epoch
            and step < self.resume_step
        )
        return skip_condition

    def _save_checkpoint(self) -> None:
        if self.checkpointing_steps is None or self.checkpoints_output_dir is None:
            return

        if (
            self.accelerator.is_main_process
            and self.global_step % self.checkpointing_steps == 0
        ):
            if self.checkpoints_total_limit is not None:

                if not os.path.exists(self.checkpoints_output_dir):
                    os.makedirs(self.checkpoints_output_dir)

                checkpoints = os.listdir(self.checkpoints_output_dir)
                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))

                if len(checkpoints) >= self.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - self.checkpoints_total_limit - 1
                    removing_checkpoints = checkpoints[:num_to_remove]

                    for rem_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(
                            self.checkpoints_output_dir, rem_checkpoint
                        )
                        shutil.rmtree(removing_checkpoint)

            save_path = os.path.join(
                self.checkpoints_output_dir, f"checkpoint-{self.global_step}"
            )
            self.accelerator.save_state(save_path)

    def tracker_evaluate_images(
        self,
        pipeline: Pix2PixColorizerPipeline,
        val_images: torch.Tensor | None = None,
        val_steps: int | None = None,
    ) -> None:

        is_valid_step = (
            self.global_step % self.eval_steps == 0
            or self.global_step == self.max_train_steps
        )
        if (
            self.accelerator.is_main_process
            and val_images is not None
            and val_steps is not None
            and is_valid_step
        ):
            for tracker in self.accelerator.trackers:
                if tracker.name == "tensorboard":
                    writer = tracker.writer
                    pred_images = pipeline(val_images)
                    grid = make_grid(pred_images, nrow=4, padding=1)
                    writer.add_images("eval-images", grid, global_step=self.global_step)

    def train(
        self,
        validation_images: torch.Tensor | None = None,
        validation_steps: int | None = None,
        disable_progress_bar: bool = False,
    ) -> None:

        if any([validation_images, validation_steps]) and not all(
            [validation_images, validation_steps]
        ):
            raise TypeError(
                "Invalid keyword arguments: "
                "Both validation_images and validation_steps must be set to log images"
            )

        self.ema.to(self.accelerator.device)
        progress_bar = tqdm(
            range(self.global_step, self.max_train_steps),
            disable=not self.accelerator.is_local_main_process or disable_progress_bar,
        )
        progress_bar.set_description("Steps")

        for _epoch in range(self.first_epoch, self.epochs):
            self.unet.train()
            train_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):

                if self._skip_step(step):
                    if step % self.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with self.accelerator.accumulate(self.unet):

                    images = batch["image"]
                    batch_size = images.shape[0]

                    light_ch, ab_ch = utils.rgb_to_zero_centered_normalized_lab(
                        images=images, split_light_and_color=True
                    )

                    noise = torch.randn(ab_ch.shape, device=self.accelerator.device)
                    timesteps = torch.randint(
                        low=0,
                        high=self.noise_scheduler.config.num_train_timesteps,
                        size=(batch_size,),
                        dtype=torch.int64,
                        device=self.accelerator.device,
                    )
                    noisy_ab_ch = self.noise_scheduler.add_noise(
                        ab_ch, noise, timesteps
                    )
                    noisy_lab_images = torch.concat([light_ch, noisy_ab_ch], dim=1)

                    noise_pred, *_ = self.unet(
                        noisy_lab_images, timesteps, return_dict=False
                    )
                    loss = F.mse_loss(noise_pred, noise)
                    avg_loss = self.accelerator.gather(loss.repeat(batch_size)).mean()
                    train_loss += avg_loss.item() / self.gradient_accumulation_steps

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)

                    self.optimizer.step()
                    if self.lr_scheduler:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    self.ema.step(self.unet.parameters())
                    progress_bar.update(1)
                    self.global_step += 1
                    self.accelerator.log(
                        values={"train_loss": train_loss}, step=self.global_step
                    )
                    train_loss = 0.0
                    self._save_checkpoint()
                progress_bar.set_postfix(step_loss=loss.detach().item())

                if self.global_step >= self.max_train_steps:
                    break

                if self.accelerator.is_main_process:
                    pipeline = Pix2PixColorizerPipeline(
                        unet=self.accelerator.unwrap_model(self.unet),
                        scheduler=self.noise_scheduler,
                    )
                    self.tracker_evaluate_images(
                        pipeline=pipeline,
                        val_images=validation_images,
                        val_steps=validation_steps,
                    )

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
