import os
import random

import torch
from PIL import Image
from skimage import io
import cv2
import numpy as np
from torch.utils.data import Dataset

from utils import pad_image, resize_image
import config


def process_real_image(image_path, transform):
    image = io.imread(image_path)
    if len(image.shape) > 2:
        if image.shape[2] > 3:
            image = image[:, :, :3]
        # image, _ = cv2.decolor(image)
        image = np.average(image, axis=2).astype(np.uint8)
    image = resize_image(image)
    image = pad_image(
        image,
        output_size=(config.IMAGE_SIZE, config.IMAGE_SIZE, 1),
    )
    image = Image.fromarray(image)
    image = transform(image)
    return torch.unsqueeze(image, dim=0)


class MakeDataset(Dataset):
    def __init__(
        self,
        input_dir,
        target_dir,
        transform_input=None,
        transform_target=None,
        shuffle=True,
    ):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.input_list_files = os.listdir(self.input_dir)
        if shuffle:
            random.shuffle(self.input_list_files)
        self.target_list_files = [
            file.replace("_bw", "_colored") for file in self.input_list_files
        ]

    def __len__(self):
        # Assuming both input and target have the same length
        return len(self.input_list_files)

    def __getitem__(self, index):
        input_img_file = self.input_list_files[index]
        input_img_path = os.path.join(self.input_dir, input_img_file)
        input_image = Image.open(input_img_path)
        input_image = pad_image(
            input_image,
            output_size=(config.IMAGE_PAD_SIZE, config.IMAGE_PAD_SIZE, 1),
            ip_type="PIL",
        )
        if self.transform_input:
            input_image = self.transform_input(input_image)

        target_img_file = self.target_list_files[index]
        target_img_path = os.path.join(self.target_dir, target_img_file)
        target_image = Image.open(target_img_path)
        target_image = pad_image(
            target_image,
            output_size=(config.IMAGE_PAD_SIZE, config.IMAGE_PAD_SIZE, 3),
            ip_type="PIL",
        )
        if self.transform_target:
            target_image = self.transform_target(target_image)

        return input_image, target_image
