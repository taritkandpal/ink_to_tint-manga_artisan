#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to fine-tune Stable Diffusion for InstructPix2Pix.
Reference: https://github.com/ShivamShrirao/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix.py
"""

import logging
import math
import json
import warnings

warnings.filterwarnings("ignore")

import datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInstructPix2PixPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler

from config import *
from utils import pad_image_diffusion, pad_image, resize_image

logger = get_logger(__name__, log_level="INFO")


def convert_to_np(image, resolution):
    """
    Helper func for preprocessing images. Converts images to RGB and resizes them.
    """
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def preprocess_images(
    examples, train_transforms, original_image_column, edited_image_column
):
    """
    Preprocess images for training dataset.
    """
    # read and process input image
    original_images = np.concatenate(
        [
            convert_to_np(
                Image.open(os.path.join(DIFFUSION_POOL_DATA, image)),
                DIFFUSION_IMAGE_SIZE,
            )
            for image in examples[original_image_column]
        ]
    )
    # read and process target image
    edited_images = np.concatenate(
        [
            convert_to_np(
                Image.open(os.path.join(DIFFUSION_POOL_DATA, image)),
                DIFFUSION_IMAGE_SIZE,
            )
            for image in examples[edited_image_column]
        ]
    )
    # We need to ensure that the input and the target images undergo the same augmentation transforms.
    images = np.concatenate([original_images, edited_images])
    images = torch.tensor(images)
    return train_transforms(images)


def collate_fn(examples):
    """
    Collate Function for Dataloader.
    """
    original_pixel_values = torch.stack(
        [example["original_pixel_values"] for example in examples]
    )
    original_pixel_values = original_pixel_values.to(
        memory_format=torch.contiguous_format
    ).float()
    edited_pixel_values = torch.stack(
        [example["edited_pixel_values"] for example in examples]
    )
    edited_pixel_values = edited_pixel_values.to(
        memory_format=torch.contiguous_format
    ).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {
        "original_pixel_values": original_pixel_values,
        "edited_pixel_values": edited_pixel_values,
        "input_ids": input_ids,
    }


def validation_fetch_data():
    """
    Function to preprocess and fetch data for validation.
    """
    # read and iterate metadata file
    metadata_fp = os.path.join(DIFFUSION_VAL_DATA, "metadata.jsonl")
    with open(metadata_fp, "r") as fh:
        metadata = [json.loads(jline) for jline in fh.read().splitlines()]
    val_data = []
    for cnt, ddict in enumerate(metadata):
        if cnt >= DIFFUSION_NUM_VAL_IMAGES:
            break
        prompt = ddict[DIFFUSION_EDIT_PROMPT_COL]
        im_fp = ddict[DIFFUSION_ORIGINAL_IMAGE_COL]
        image = Image.open(os.path.join(DIFFUSION_VAL_DATA, im_fp)).convert("RGB")
        image = np.array(image)
        # resize and pad image
        image = resize_image(image)
        image = pad_image(image, (DIFFUSION_IMAGE_SIZE, DIFFUSION_IMAGE_SIZE, 3))
        image = Image.fromarray(image)
        val_data.append((image, prompt))
    return val_data


def main():
    """
    Main function for training stable diffusion.
    """
    diffusion_train_epochs = DIFFUSION_TRAIN_EPOCHS

    logging_dir = os.path.join(DIFFUSION_OUTPUT_DIR, "logs")

    # set configs and initialize accelerator
    accelerator_project_config = ProjectConfiguration(
        logging_dir=logging_dir,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=DIFFUSION_GRAD_ACC_STEPS,
        mixed_precision=DIFFUSION_MIXED_PRECISION,
        log_with="tensorboard",
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    generator = torch.Generator(device=accelerator.device).manual_seed(SEED)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(SEED)

    # Handle the repository creation
    if accelerator.is_main_process:
        if DIFFUSION_OUTPUT_DIR is not None:
            os.makedirs(DIFFUSION_OUTPUT_DIR, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        DIFFUSION_MODEL, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        DIFFUSION_MODEL,
        subfolder="tokenizer",
        revision=DIFFUSION_MODEL_REVISION,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        DIFFUSION_MODEL,
        subfolder="text_encoder",
        revision=DIFFUSION_MODEL_REVISION,
    )
    vae = AutoencoderKL.from_pretrained(
        DIFFUSION_MODEL,
        subfolder="vae",
        revision=DIFFUSION_MODEL_REVISION,
    )
    unet = UNet2DConditionModel.from_pretrained(
        DIFFUSION_MODEL,
        subfolder="unet",
        revision=DIFFUSION_MODEL_REVISION,
    )

    # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
    # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet. This UNet is
    # then fine-tuned on the custom InstructPix2Pix dataset. This modified UNet is initialized
    # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
    # initialized to zero.
    if accelerator.is_main_process:
        logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
        in_channels = 8
        out_channels = unet.conv_in.out_channels
        unet.register_to_config(in_channels=in_channels)

        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels,
                out_channels,
                unet.conv_in.kernel_size,
                unet.conv_in.stride,
                unet.conv_in.padding,
            )
            new_conv_in.weight.zero_()
            # new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            new_conv_in.weight[:, :in_channels, :, :].copy_(unet.conv_in.weight)
            unet.conv_in = new_conv_in

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        """
        Custom saving hook.
        """
        for i, model in enumerate(models):
            model.save_pretrained(os.path.join(output_dir, "unet"))
            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(models, input_dir):
        """
        Custom loading hook.
        """
        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()
            # load diffusers style into model
            load_model = UNet2DConditionModel.from_pretrained(
                input_dir, subfolder="unet"
            )
            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=DIFFUSION_ADAM_LR,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-08,
    )

    # Get the datasets
    data_files = dict()
    data_files["train"] = os.path.join(DIFFUSION_TRAIN_DATA, "**")
    dataset = load_dataset(
        "imagefolder",
        data_files=data_files,
        drop_labels=True,
    )

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    original_image_column = DIFFUSION_ORIGINAL_IMAGE_COL
    if original_image_column not in column_names:
        raise ValueError(
            f"original_image_column' value '{DIFFUSION_ORIGINAL_IMAGE_COL}' needs to be one of: {', '.join(column_names)}"
        )
    edit_prompt_column = DIFFUSION_EDIT_PROMPT_COL
    if edit_prompt_column not in column_names:
        raise ValueError(
            f"edit_prompt_column' value '{DIFFUSION_EDIT_PROMPT_COL}' needs to be one of: {', '.join(column_names)}"
        )
    edited_image_column = DIFFUSION_EDITED_IMAGE_COL
    if edited_image_column not in column_names:
        raise ValueError(
            f"edited_image_column' value '{DIFFUSION_EDITED_IMAGE_COL}' needs to be one of: {', '.join(column_names)}"
        )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    # Transforms
    train_transforms = transforms.Compose(
        [
            transforms.Lambda(
                lambda x: pad_image_diffusion(
                    x,
                    output_size=(DIFFUSION_INPUT_SIZE, DIFFUSION_INPUT_SIZE, 3),
                    ip_type="PIL",
                )
            ),
            transforms.Resize(
                DIFFUSION_IMAGE_SIZE, transforms.InterpolationMode.BILINEAR
            ),
            transforms.Lambda(lambda x: 2 * (x / 255) - 1),
        ]
    )

    def preprocess_train(examples):
        """
        Prepare training dataset.
        """
        # Preprocess images.
        preprocessed_images = preprocess_images(
            examples, train_transforms, original_image_column, edited_image_column
        )
        # Since the original and edited images were concatenated before
        # applying the transformations, we need to separate them and reshape
        # them accordingly.
        original_images, edited_images = preprocessed_images.chunk(2)
        original_images = original_images.reshape(
            -1, 3, DIFFUSION_IMAGE_SIZE, DIFFUSION_IMAGE_SIZE
        )
        edited_images = edited_images.reshape(
            -1, 3, DIFFUSION_IMAGE_SIZE, DIFFUSION_IMAGE_SIZE
        )

        # Collate the preprocessed images into the `examples`.
        examples["original_pixel_values"] = original_images
        examples["edited_pixel_values"] = edited_images

        # Preprocess the captions.
        captions = [caption for caption in examples[edit_prompt_column]]
        examples["input_ids"] = tokenize_captions(captions)
        return examples

    with accelerator.main_process_first():
        # truncate train samples if configured
        if DIFFUSION_MAX_TRAIN_SAMPLES is not None:
            dataset["train"] = (
                dataset["train"]
                # .shuffle(seed=SEED)
                .select(range(DIFFUSION_MAX_TRAIN_SAMPLES))
            )
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=DIFFUSION_BATCH_SIZE,
        num_workers=0,
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / DIFFUSION_GRAD_ACC_STEPS
    )
    max_train_steps = diffusion_train_epochs * num_update_steps_per_epoch

    # Learning Rate Scheduler
    lr_scheduler = get_scheduler(
        DIFFUSION_LR_SCHEDULER,
        optimizer=optimizer,
        num_warmup_steps=100 * DIFFUSION_GRAD_ACC_STEPS,
        num_training_steps=max_train_steps * DIFFUSION_GRAD_ACC_STEPS,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / DIFFUSION_GRAD_ACC_STEPS
    )
    max_train_steps = diffusion_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    diffusion_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("instruct-pix2pix")

    # Train!
    total_batch_size = (
        DIFFUSION_BATCH_SIZE * accelerator.num_processes * DIFFUSION_GRAD_ACC_STEPS
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {diffusion_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {DIFFUSION_BATCH_SIZE}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {DIFFUSION_GRAD_ACC_STEPS}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, diffusion_train_epochs):
        unet.train()
        train_loss = 0.0
        step_lrs = []
        step_losses = []
        if not DIFFUSION_INFERENCE_ONLY:
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    # We want to learn the denoising process w.r.t the edited images which
                    # are conditioned on the original image (which was edited) and the edit instruction.
                    # So, first, convert images to latent space.
                    latents = vae.encode(
                        batch["edited_pixel_values"].to(weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning.
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Get the additional image embedding for conditioning.
                    # Instead of getting a diagonal Gaussian here, we simply take the mode.
                    original_image_embeds = vae.encode(
                        batch["original_pixel_values"].to(weight_dtype)
                    ).latent_dist.mode()

                    # Concatenate the `original_image_embeds` with the `noisy_latents`.
                    concatenated_noisy_latents = torch.cat(
                        [noisy_latents, original_image_embeds], dim=1
                    )

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                        )

                    # Predict the noise residual and compute loss
                    model_pred = unet(
                        concatenated_noisy_latents, timesteps, encoder_hidden_states
                    ).sample
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(
                        loss.repeat(DIFFUSION_BATCH_SIZE)
                    ).mean()
                    train_loss += avg_loss.item() / DIFFUSION_GRAD_ACC_STEPS

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), 1)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                step_losses.append(logs["step_loss"])
                step_lrs.append(logs["lr"])
                progress_bar.set_postfix(**logs)

                if global_step >= max_train_steps:
                    break

            # plot loss curve and lr curve
            os.makedirs(DIFFUSION_VALIDATION_OUTPUT_DIR, exist_ok=True)
            plt.plot(list(range(len(step_losses))), step_losses)
            plt.title("Train Loss")
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.savefig(
                os.path.join(DIFFUSION_VALIDATION_OUTPUT_DIR, f"loss_curve.png")
            )
            plt.cla()
            plt.clf()
            plt.plot(list(range(len(step_lrs))), step_lrs)
            plt.title("Learning Rate")
            plt.xlabel("Steps")
            plt.ylabel("LR")
            plt.savefig(os.path.join(DIFFUSION_VALIDATION_OUTPUT_DIR, f"lr_curve.png"))
            plt.cla()
            plt.clf()

            # Create the pipeline using the trained modules and save it.
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unet = accelerator.unwrap_model(unet)
                pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    DIFFUSION_MODEL,
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    vae=accelerator.unwrap_model(vae),
                    unet=unet,
                    revision=DIFFUSION_MODEL_REVISION,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
                pipeline.save_pretrained(DIFFUSION_OUTPUT_DIR)

        if accelerator.is_main_process:
            if epoch % DIFFUSION_VAL_EPOCHS == 0:
                # Run generation on validation data
                logger.info(
                    f"Running validation. Generating {DIFFUSION_NUM_VAL_IMAGES} images."
                )
                # create pipeline
                pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    DIFFUSION_MODEL,
                    unet=unet,
                    revision=DIFFUSION_MODEL_REVISION,
                    torch_dtype=weight_dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                val_data = validation_fetch_data()
                edited_images = []
                with torch.autocast(
                    str(accelerator.device),
                    enabled=accelerator.mixed_precision == "fp16",
                ):
                    cnt = 0
                    for original_image, validation_prompt in tqdm(
                        val_data, desc="Validation"
                    ):
                        image = pipeline(
                            validation_prompt,
                            image=original_image,
                            num_inference_steps=DIFFUSION_NUM_INFERENCE_STEPS,
                            image_guidance_scale=DIFFUSION_IMAGE_GUIDANCE_SCALE,
                            guidance_scale=DIFFUSION_GUIDANCE_SCALE,
                            generator=generator,
                            negative_prompt="Color",
                        ).images[0]
                        edited_images.append(image)
                        image.save(
                            os.path.join(DIFFUSION_VALIDATION_OUTPUT_DIR, f"{cnt}.png")
                        )
                        cnt += 1

                del pipeline
                torch.cuda.empty_cache()

    accelerator.end_training()


if __name__ == "__main__":
    main()
