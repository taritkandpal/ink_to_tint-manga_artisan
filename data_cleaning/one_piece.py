"""
One-time script for preparing OnePiece manga images.
"""

import os
import multiprocessing as mp
from pathlib import Path

from skimage import io
from tqdm import tqdm

from utils import image_filter_pixels, pdf_to_image, resize_image
from config import *


ip_path = Path("Datasets/StyleChange/Raw/OnePiece")
int_path = Path("Datasets/StyleChange/Intermediate/OnePiece")
op_path = Path("Datasets/StyleChange/Final/OnePiece")
os.makedirs(int_path, exist_ok=True)
os.makedirs(op_path, exist_ok=True)

wr_lb = 0.20347271516234758
wr_ub = 0.34679431575579284
br_lb = 0.01980532019161006
br_ub = 0.09066416666666667

onepiece_folders = [
    ip_path / dpath / "One Piece Manga" for dpath in os.listdir(ip_path)
]


def find_files():
    files_list = []
    fnames = {}
    for folder in onepiece_folders:
        files = os.listdir(folder)
        for file in files:
            if file not in fnames:
                fnames[file] = []
            ip_file = str(folder / file)
            renamed_file = "".join(
                file.split(".")[:-1] + [f"_{len(fnames[file])}"] + ["_{}.png"]
            ).replace(" ", "")
            files_list.append((ip_file, renamed_file))
            fnames[file].append(ip_file)
    return files_list


def process_pdf(fpaths):
    try:
        ip_file_path, op_file = fpaths
        pdf_to_image(ip_file_path, int_path, op_file, zoom=1)
        return 1
    except Exception as e:
        print(e)
        return 0


def process_file(ip_image_name):
    try:
        white_ratio, black_ratio, rgb, manga_ar, image = image_filter_pixels(
            int_path / ip_image_name, return_image=True
        )
        if rgb:
            return 2
        if not manga_ar:
            return 3
        if (
            not rgb
            and manga_ar
            and wr_lb < white_ratio < wr_ub
            and br_lb < black_ratio < br_ub
        ):
            image = resize_image(image)
            io.imsave(op_path / ip_image_name, image)
            return 1
        else:
            return 4
    except Exception as e:
        print(e)
        return 0


if __name__ == "__main__":
    # pdf to images
    files_list = find_files()
    with mp.Pool(16) as p:
        op = list(
            tqdm(
                p.imap_unordered(process_pdf, files_list),
                total=len(files_list),
                desc="Processing OnePiece PDFs",
            )
        )
    print(f"{op.count(1)} pdfs processed successfully. {op.count(0)} pdfs failed.")

    # image processing
    unprocessed_files = os.listdir(int_path)
    with mp.Pool(12) as p:
        op = list(
            tqdm(
                p.imap_unordered(process_file, unprocessed_files),
                total=len(unprocessed_files),
                desc="Processing OnePiece images",
            )
        )

    print(
        f"{op.count(1)} images processed successfully. {op.count(0)} images failed. {op.count(2)} were rgb images. "
        f"{op.count(3)} were of incorrect aspect ratio. {op.count(4)} had invalid white/black ratios."
    )


# Processing OnePiece PDFs: 100%|██████████| 1020/1020 [05:52<00:00,  2.90it/s]
# 1020 pdfs processed successfully. 0 pdfs failed.
