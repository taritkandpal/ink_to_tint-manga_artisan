"""
One-time script for preparing MobPsycho manga images.
"""

import os
import multiprocessing as mp
from pathlib import Path

from skimage import io
from tqdm import tqdm

from utils import resize_image, check_aspect_ratio, check_rgb
from config import *

ip_path = Path(
    "Datasets/StyleChange/Raw/MobPsycho/Mob Psycho 100 Manga-20231118T024328Z-001/Mob Psycho 100 Manga/01 Main Series (individual pages)"
)
op_path = Path("Datasets/StyleChange/Final/MobPsycho")
os.makedirs(op_path, exist_ok=True)


def process_image(im_tup):
    try:
        ip_image_path, op_image_name = im_tup
        image = io.imread(ip_image_path)
        if not check_aspect_ratio(image) or check_rgb(image):
            return 2
        image = resize_image(image)
        io.imsave(op_path / op_image_name, image)
        return 1
    except Exception as e:
        print(e)
        return 0


if __name__ == "__main__":
    fset = []
    fnames = {}
    for root, dirs, files in os.walk(ip_path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                if file not in fnames:
                    fnames[file] = []
                renamed_file = "".join(
                    file.split(".")[:-1] + [f"_{len(fnames[file])}.png"]
                )
                fset.append((os.path.join(root, file), renamed_file))
                fnames[file].append(renamed_file)

    with mp.Pool(16) as p:
        op = list(
            tqdm(
                p.imap_unordered(process_image, fset),
                total=len(fset),
                desc="Processing mob_psycho images",
            )
        )
    print(
        f"{op.count(1)} images processed successfully. {op.count(0)} images failed. "
        f"{op.count(2)} were rgb images or of incorrect aspect ratio. "
    )

# Processing mob_psycho images:  49%|████▉     | 1594/3255 [00:07<00:02, 629.54it/s]C:\Assignments\Computational_Imaging\project\data_cleaning\mob_psycho.py:29: UserWarning: C:\Assignments\Computational_Imaging\project\Datasets\StyleChange\Final\MobPsycho\img000004_11.png is a low contrast image
# Processing mob_psycho images: 100%|██████████| 3255/3255 [00:11<00:00, 276.07it/s]
# 1999 images processed successfully. 0 images failed. 1256 were rgb images or of incorrect aspect ratio.
