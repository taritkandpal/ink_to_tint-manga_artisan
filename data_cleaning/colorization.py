import os
import multiprocessing as mp
from pathlib import Path
import warnings

from tqdm import tqdm
from skimage import io
import cv2
import numpy as np

from utils import (
    resize_image,
    check_aspect_ratio,
    check_rgb,
    image_filter_pixels,
    pad_image,
    decolorize_image,
)
from config import *


warnings.filterwarnings("ignore")

ip_path = Path("Datasets/Colorization/archive/color_full")
colored_op_path = Path("Datasets/Colorization/Final/coloured")
bw_op_path = Path("Datasets/Colorization/Final/bw")
os.makedirs(colored_op_path, exist_ok=True)
os.makedirs(bw_op_path, exist_ok=True)

ip_image_names = os.listdir(ip_path)


def temp():
    ip = r"Datasets/Colorization/Debug_Raw/train/coloured"
    op = r"Datasets/Colorization/Debug/train/coloured"
    for file in os.listdir(ip):
        image = io.imread(os.path.join(ip, file))
        op_size = (600, 600, 3) if len(image.shape) == 3 else (600, 600)
        image = pad_image(image, op_size)
        io.imsave(os.path.join(op, file), image)


def white_pixels_fix(colored_image, bw_image):
    colored_image = np.average(colored_image, axis=2).astype(np.uint8)
    comp = np.full(colored_image.shape, 255, np.uint8)
    filler = comp == colored_image
    bw_image[filler] = 255
    return bw_image


def process_image(ip_image_name):
    try:
        image = io.imread(ip_path / ip_image_name)
        if image.shape[2] > 3:
            image = image[:, :, :3]
        if not check_aspect_ratio(image) or not check_rgb(image):
            return 0
        bw_image = decolorize_image(image)
        colored_image = resize_image(image)
        bw_image = resize_image(bw_image)
        colored_image_name = "".join(ip_image_name.split(".") + ["_colored.png"])
        # bw_image, _ = cv2.decolor(colored_image)
        # bw_image = np.average(colored_image, axis=2).astype(np.uint8)
        # bw_image = white_pixels_fix(colored_image, bw_image)
        bw_image_name = "".join(ip_image_name.split(".") + ["_bw.png"])
        white_ratio, black_ratio, rgb, manga_ar = image_filter_pixels(bw_image)
        if 0.1 < white_ratio < 0.9:
            io.imsave(colored_op_path / colored_image_name, colored_image)
            io.imsave(bw_op_path / bw_image_name, bw_image)
            return 1
        return 0
    except Exception as e:
        # print(e)
        return 0


if __name__ == "__main__":
    with mp.Pool(16) as p:
        op = list(
            tqdm(
                p.imap_unordered(process_image, ip_image_names),
                total=len(ip_image_names),
                desc="Processing coloured images",
            )
        )

    # op = []
    # for ip_image_name in tqdm(ip_image_names, desc="Processing coloured images"):
    #     op.append(process_image(ip_image_name))

    print(f"{op.count(1)} images processed successfully. {op.count(0)} images failed.")


# Processing coloured images:  97%|█████████▋| 56954/58642 [31:01<00:41, 40.59it/s]OpenCV(4.8.1) D:\a\opencv-python\opencv-python\opencv\modules\photo\src\contrast_preserve.cpp:64: error: (-215:Assertion failed) !I.empty() && (I.channels()==3) in function 'cv::decolor'
# Processing coloured images:  97%|█████████▋| 57034/58642 [31:03<00:50, 31.92it/s]too many indices for array: array is 2-dimensional, but 3 were indexed
# Processing coloured images:  99%|█████████▉| 57989/58642 [31:34<00:18, 36.19it/s]Could not find a backend to open `C:\Assignments\Computational_Imaging\project\Datasets\Colorization\archive\color_full\Color 9902878163859763.jpg`` with iomode `r`.
# Based on the extension, the following plugins might add capable backends:
#   pyav:  pip install imageio[pyav]
# Processing coloured images: 100%|██████████| 58642/58642 [31:53<00:00, 30.64it/s]
# 50607 images processed successfully. 8035 images failed.

# Processing coloured images: 100%|██████████| 58642/58642 [06:38<00:00, 147.11it/s]
# 50782 images processed successfully. 7860 images failed.

# Processing coloured images: 100%|██████████| 58642/58642 [09:12<00:00, 106.19it/s]
# 49513 images processed successfully. 9129 images failed.
