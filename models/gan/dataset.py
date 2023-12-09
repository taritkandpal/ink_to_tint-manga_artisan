"""
Class definitions for GAN Dataset.
"""
import os
import random

import torch
from PIL import Image
from skimage import io
import numpy as np
from torch.utils.data import Dataset

from utils import pad_image, resize_image
import config


def process_real_image(image_path, transform=None):
    """
    Function to process unseen / real images during validation.
    """
    image = io.imread(image_path)
    # if black-and-white is read as coloured then convert to single channel
    if len(image.shape) > 2:
        # if png image has 4th alpha channel then remove it
        if image.shape[2] > 3:
            image = image[:, :, :3]
        image = np.average(image, axis=2).astype(np.uint8)
    # resize image
    image = resize_image(image)
    # pad image
    image = pad_image(
        image,
        output_size=(config.IMAGE_SIZE, config.IMAGE_SIZE, 1),
    )
    # convert to PIL image and apply transforms
    image = Image.fromarray(image)
    if transform:
        image = transform(image)
    return torch.unsqueeze(image, dim=0)


class MakeDataset(Dataset):
    """
    Custom Dataset class for GAN images.
    """

    def __init__(
        self,
        input_dir,
        target_dir,
        transform_input=None,
        transform_target=None,
        shuffle=True,
    ):
        # directories
        self.input_dir = input_dir
        self.target_dir = target_dir
        # transforms
        self.transform_input = transform_input
        self.transform_target = transform_target
        # filepaths
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
        """
        Fetch item at index from dataset.
        """
        # read input image
        input_img_file = self.input_list_files[index]
        input_img_path = os.path.join(self.input_dir, input_img_file)
        input_image = Image.open(input_img_path)
        # pad input image
        input_image = pad_image(
            input_image,
            output_size=(config.IMAGE_PAD_SIZE, config.IMAGE_PAD_SIZE, 1),
            ip_type="PIL",
        )
        # apply transforms to input image
        if self.transform_input:
            input_image = self.transform_input(input_image)

        # read target image
        target_img_file = self.target_list_files[index]
        target_img_path = os.path.join(self.target_dir, target_img_file)
        target_image = Image.open(target_img_path)
        # pad target image
        target_image = pad_image(
            target_image,
            output_size=(config.IMAGE_PAD_SIZE, config.IMAGE_PAD_SIZE, 3),
            ip_type="PIL",
        )
        # apply transforms to target image
        if self.transform_target:
            target_image = self.transform_target(target_image)

        return input_image, target_image
