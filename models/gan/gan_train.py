import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

from utils import save_checkpoint, save_image, load_checkpoint
import config
from models.gan.dataset import MakeDataset, process_real_image
from models.gan.generator import Generator
from models.gan.discriminator import Discriminator


torch.manual_seed(config.SEED)
random.seed(config.SEED)
np.random.seed(config.SEED)
os.chdir(config.PROJECT_PATH)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def save_real_images(gen=None, transform=None):
    if not gen:
        gen = Generator(in_channels=1).to(config.DEVICE)
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
        )
    if not transform:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    config.IMAGE_SIZE, transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[config.MEAN], std=[config.STD]),
            ]
        )
    op_folder = os.path.join(config.TEST_FOLDER, "Real")
    gen.eval()
    os.makedirs(op_folder, exist_ok=True)
    for im_name in os.listdir(config.REAL_IMAGES):
        image = process_real_image(os.path.join(config.REAL_IMAGES, im_name), transform)
        image = image.to(config.DEVICE)
        with torch.no_grad():
            pred = gen(image)
        pred = pred * config.STD + config.MEAN
        save_image(pred, os.path.join(op_folder, im_name))
    gen.train()


def save_some_examples(gen, test_loader, epoch):
    gen.eval()
    op_folder = os.path.join(config.TEST_FOLDER, str(epoch))
    os.makedirs(op_folder, exist_ok=True)
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        with torch.no_grad():
            y_fake = gen(x)
            y_fake = y_fake * config.STD + config.MEAN  # remove normalization
            x = x * config.STD + config.MEAN
            y = y * config.STD + config.MEAN
            save_image(y_fake, os.path.join(op_folder, f"{epoch}_generated_{i}.png"))
            if epoch == 0:
                save_image(x, os.path.join(op_folder, f"input_{i}.png"))
                save_image(y, os.path.join(op_folder, f"label_{i}.png"))
    gen.train()


def train_model(
    disc,
    gen,
    dataloader,
    optimizer_disc,
    optimizer_gen,
    # scheduler_disc,
    # scheduler_gen,
    l1_loss,
    bce,
    gen_scaler,
    disc_scaler,
):
    loop = tqdm(dataloader)
    for index, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        # TRAIN DISCRIMINATOR
        # Automatically casts certain operations to a lower precision (like float16) to speed up computation
        # and reduce memory usage during forward and backward passes.
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())
            # compares single valued D_real value in a batch with 1. So if D_real is 0.9, then,
            # D_real is actually classified as real image
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        optimizer_disc.zero_grad()
        disc_scaler.scale(D_loss).backward()
        disc_scaler.step(optimizer_disc)
        disc_scaler.update()
        # D_loss.backward()
        # optimizer_disc.step()
        # scheduler_disc.step()

        # TRAIN GENERATOR
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        optimizer_gen.zero_grad()
        gen_scaler.scale(G_loss).backward()
        gen_scaler.step(optimizer_gen)
        gen_scaler.update()
        # G_loss.backward()
        # optimizer_gen.step()
        # scheduler_gen.step()

        # To calculate the average confidence of the discriminator in it's real/fake images
        # every 10 batches to check if the discriminator is performing well during training.
        # Eg: D_real value should be close to 1 and D_fake value should be close to 0
        if index % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                D_loss=D_loss.item(),
                G_loss=G_loss.item(),
            )


def main():
    transform_input = transforms.Compose(
        [
            # transforms.Resize(config.IMAGE_SIZE, transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[config.MEAN], std=[config.STD]),
        ]
    )
    transform_target = transforms.Compose(
        [
            # transforms.Resize(config.IMAGE_SIZE, transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
        ]
    )
    disc = Discriminator(in_channels=4).to(config.DEVICE)
    gen = Generator(in_channels=1).to(config.DEVICE)
    optimizer_disc = optim.AdamW(
        disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)
    )
    optimizer_gen = optim.AdamW(
        gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999)
    )
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    # Model loads weights from previously saved checkpoints
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            optimizer_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC,
            disc,
            optimizer_disc,
            config.LEARNING_RATE,
        )

    dataset = MakeDataset(
        config.INPUT_DIR, config.TARGET_DIR, transform_input, transform_target
    )

    if config.TRAIN_RATIO:
        train_size = int(config.TRAIN_RATIO * len(dataset))
        test_size = len(dataset) - train_size
    else:
        train_size = len(dataset) - config.NUM_SAMPLES_PER_EPOCH
        test_size = config.NUM_SAMPLES_PER_EPOCH

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    gen_scaler = torch.cuda.amp.GradScaler()
    disc_scaler = torch.cuda.amp.GradScaler()

    # scheduler_disc = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer_disc, T_max=config.NUM_EPOCHS * len(train_loader), eta_min=config.MIN_LR
    # )
    # scheduler_gen = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer_gen, T_max=config.NUM_EPOCHS * len(train_loader), eta_min=config.MIN_LR
    # )

    # run epochs
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nStarting epoch {epoch}.")
        train_model(
            disc,
            gen,
            train_loader,
            optimizer_disc,
            optimizer_gen,
            # scheduler_disc,
            # scheduler_gen,
            L1_LOSS,
            BCE,
            gen_scaler,
            disc_scaler,
        )

        save_checkpoint(disc, optimizer_disc, filename=config.CHECKPOINT_DISC)
        save_checkpoint(gen, optimizer_gen, filename=config.CHECKPOINT_GEN)

        save_some_examples(gen, test_loader, epoch)
    save_real_images(gen)


if __name__ == "__main__":
    main()
