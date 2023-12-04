import gc
import os
import multiprocessing as mp
from pathlib import Path

from skimage import io
from tqdm import tqdm

from utils import image_filter_pixels, pdf_to_image, resize_image
from config import *

ip_path = Path("Datasets/StyleChange/Raw/Jojo")
int_path = Path("Datasets/StyleChange/Intermediate/Jojo")
op_path = Path("Datasets/StyleChange/Final/Jojo")
os.makedirs(int_path, exist_ok=True)
os.makedirs(op_path, exist_ok=True)

wr_lb = 0.28386610477727037
wr_ub = 0.5514879389194435

jojo_folders = [
    ip_path / dpath for dpath in os.listdir(ip_path)
]


def find_files():
    files_list = []
    fnames = {}
    for folder in jojo_folders:
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
        pdf_to_image(ip_file_path, int_path, op_file, zoom=3)
        return 1
    except Exception as e:
        print(e)
        return 0


def process_file(ip_image_name):
    try:
        gc.collect()
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
    # files_list = find_files()
    # with mp.Pool(16) as p:
    #     op = list(
    #         tqdm(
    #             p.imap_unordered(process_pdf, files_list),
    #             total=len(files_list),
    #             desc="Processing Jojo PDFs",
    #         )
    #     )
    # print(f"{op.count(1)} pdfs processed successfully. {op.count(0)} pdfs failed.")

    # image processing
    unprocessed_files = os.listdir(int_path)
    with mp.Pool(12) as p:
        op = list(
            tqdm(
                p.imap_unordered(process_file, unprocessed_files),
                total=len(unprocessed_files),
                desc="Processing Jojo images",
            )
        )
    print(
        f"{op.count(1)} images processed successfully. {op.count(0)} images failed. {op.count(2)} were rgb images. "
        f"{op.count(3)} were of incorrect aspect ratio. {op.count(4)} had invalid white/black ratios."
    )


# Processing Jojo PDFs: 100%|██████████| 64/64 [03:25<00:00,  3.21s/it]
# 64 pdfs processed successfully. 0 pdfs failed.
# Processing Jojo images: 100%|██████████| 10536/10536 [03:03<00:00, 57.49it/s]
# 4731 images processed successfully. 0 images failed. 1107 were rgb images. 1544 were of incorrect aspect ratio. 3154 had invalid white/black ratios.
