"""
One-time script for preparing Naruto manga images.
"""

import os
from pathlib import Path
from tqdm import tqdm
from skimage import io

from utils import image_filter_pixels, resize_image
from config import *


ip_path = Path("Datasets/StyleChange/Raw/Naruto")
op_path = Path("Datasets/StyleChange/Final/Naruto")
os.makedirs(op_path, exist_ok=True)

naruto_folders = [
    ip_path / dpath / "Naruto Manga - [Volumes]" for dpath in os.listdir(ip_path)
]
files_list = []
fnames = {}
for naruto_folder in naruto_folders:
    volumes = [naruto_folder / volume for volume in os.listdir(naruto_folder)]
    for volume in tqdm(volumes):
        files = os.listdir(volume)
        for file in files:
            if file not in fnames:
                fnames[file] = []
            ip_file = volume / file
            renamed_file = "".join(
                file.split(".")[:-1] + [f"_{len(fnames[file])}"] + [".png"]
            )
            op_file = op_path / renamed_file
            files_list.append((ip_file, op_file))
            fnames[file].append(ip_file)

counter = 0
rgb_counter = 0
manga_ar_counter = 0
wr_lb = 0.2549088322584268
wr_ub = 0.528454450004755
br_lb = 0.003620426829268293
br_ub = 0.06131591730987208
for ip_file, op_file in tqdm(files_list, desc="Processing Naruto files"):
    white_ratio, black_ratio, rgb, manga_ar = image_filter_pixels(ip_file)
    if rgb:
        rgb_counter += 1
    if not manga_ar:
        manga_ar_counter += 1
    if (
        not rgb
        and manga_ar
        and wr_lb < white_ratio < wr_ub
        and br_lb < black_ratio < br_ub
    ):
        image = io.imread(ip_file)
        image = resize_image(image)
        io.imsave(op_file, image)
        counter += 1
print(
    f"{counter} images copied successfully. RGB count: {rgb_counter}. Manga Invalid AR count: {manga_ar_counter}"
)

# 100%|██████████| 68/68 [00:00<00:00, 2956.49it/s]
# 100%|██████████| 63/63 [00:00<00:00, 2737.29it/s]
# 100%|██████████| 51/51 [00:00<00:00, 2484.83it/s]
# 100%|██████████| 37/37 [00:00<00:00, 1783.29it/s]
# 100%|██████████| 18/18 [00:00<00:00, 750.17it/s]
# Processing Naruto files: 100%|██████████| 13577/13577 [12:42<00:00, 17.79it/s]
# 2932 images copied successfully. RGB count: 3518. Manga Invalid AR count: 395
