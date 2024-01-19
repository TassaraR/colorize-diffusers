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
from tqdm.auto import tqdm

import colorizer.utils as utils


class Trainer:
    def __init__(
        self,
        unet: UNet2DModel,
        noise_scheduler: DDPMScheduler,
        ema: EMAModel,
        optimizer: Optimizer,
        train_dataloader,
        epochs,
        gradient_accumulation_steps,
        checkpointing_steps,
        checkpoints_total_limit,
        checkpoints_output_dir,
        max_train_steps: int | None = None,
        lr_scheduler: Callable | None = None,
        accelerator_kwargs: dict | None = None,
    ):
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.train_dataloader = train_dataloader
        self.ema = ema
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.accelerator = Accelerator(**accelerator_kwargs)
        # self.accelerator.init_trackers()

        self.epochs = epochs
        self.global_step = 0
        self.first_epoch = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.checkpointing_steps = checkpointing_steps
        self.checkpoints_total_limit = checkpoints_total_limit
        self.resume_from_checkpoint = False
        self.checkpoints_output_dir = checkpoints_output_dir
        self.resume_step = -1

        # CALC TRAINING STEPS
        overrode_max_train_steps = False
        self.num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / gradient_accumulation_steps
        )
        if max_train_steps is None:
            self.max_train_steps = self.epochs * self.num_update_steps_per_epoch
            overrode_max_train_steps = True

        self.num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / gradient_accumulation_steps
        )
        if overrode_max_train_steps:
            self.max_train_steps = self.epochs * self.num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.epochs = math.ceil(self.max_train_steps / self.num_update_steps_per_epoch)

    def load_checkpoint(self, path: str):

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

    def _skip_step(self, step):
        skip_condition = (
            self.resume_from_checkpoint
            and self.epochs == self.first_epoch
            and step < self.resume_step
        )
        return skip_condition

    def _save_checkpoint(self):
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

    def train(self):

        self.ema.to(self.accelerator.device)

        (
            self.unet,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        progress_bar = tqdm(
            range(self.global_step, self.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )

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
                    train_loss = 0.0

            self._save_checkpoint()
