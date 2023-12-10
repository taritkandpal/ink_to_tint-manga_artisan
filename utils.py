from pathlib import Path, WindowsPath
import json

from skimage import io
from PIL import Image
import cv2
import fitz
import numpy as np
from tqdm import tqdm

from config import *


def ldict_to_jsonl(ldict, file_path):
    """
    Function to write list of dictionaries to jsonl file.
    """
    with open(file_path, "w") as out:
        for data in ldict:
            jout = json.dumps(data) + "\n"
            out.write(jout)


def read_jsonl(file_path):
    """
    Function to read jsonl file and return metadata.
    """
    with open(file_path, "r") as fh:
        jsonl_content = fh.read()
    metadata = [json.loads(jline) for jline in jsonl_content.splitlines()]
    return metadata


def check_rgb(image):
    """
    Function to check if 3 channel image is rgb or not.
    """
    c1 = (image[:, :, 0] == image[:, :, 1]).astype(int)
    c2 = (image[:, :, 1] == image[:, :, 2]).astype(int)
    if np.count_nonzero(c1 == 0) != 0 or np.count_nonzero(c2 == 0) != 0:
        return True
    return False


def check_aspect_ratio(image):
    """
    Function to check aspect whether aspect raio lies in expected range or not.
    """
    aspect_ratio = image.shape[0] / image.shape[1]
    return ASPECT_RATIO_LB <= aspect_ratio <= ASPECT_RATIO_UB


def pad_image(image, output_size, pad_value=0, ip_type="numpy"):
    """
    Pad image evenly in all axes to get specific output size.
    """
    if ip_type == "PIL":
        image = np.array(image)
    pad_values = [
        (max(0, (dim - original_dim) // 2), max(0, (dim - original_dim + 1) // 2))
        for dim, original_dim in zip(output_size, image.shape)
    ]
    padded_array = np.pad(
        image, pad_values, mode="constant", constant_values=pad_value
    ).astype(np.uint8)
    if ip_type == "PIL":
        padded_array = Image.fromarray(padded_array)
    return padded_array


def pad_image_diffusion(image, output_size, pad_value=0, ip_type="numpy"):
    """
    Pad concatenated group of images evenly in all axes to get specific output size.
    """
    padded_arrays = []
    for i in image:
        padded_array = pad_image(i, output_size, pad_value, ip_type)
        padded_arrays.append(
            np.array(padded_array).reshape(
                (1, DIFFUSION_INPUT_SIZE, DIFFUSION_INPUT_SIZE)
            )
        )
    a = torch.cat([torch.from_numpy(x) for x in padded_arrays])
    return a


def resize_image(image, output_size=RESIZED_IMAGE_2D_DIM, antialiasing=True):
    """
    Resize image to specific output size.
    """
    intrp = cv2.INTER_AREA if antialiasing else cv2.INTER_LINEAR
    return cv2.resize(image, output_size, interpolation=intrp).astype(np.uint8)


def decolorize_image(image):
    """
    Function to decolorize image using pencillation and bolden using dilation.
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_gray_inv = 255 - img_gray
    img_blur = cv2.bilateralFilter(img_gray_inv, d=7, sigmaColor=300, sigmaSpace=50)
    dodgeV2 = lambda img, mask: cv2.divide(img, 255 - mask, scale=256)
    img_blend = dodgeV2(img_gray, img_blur)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_blend = 255 - cv2.dilate(255 - img_blend, kernel, iterations=1)
    return img_blend


def remove_fourth_channel(data_path, output_path=None):
    """
    Function to remove 4th channel from png images.
    """
    if output_path is None:
        output_path = data_path
    files = os.listdir(data_path)
    for file in tqdm(files):
        image = io.imread(os.path.join(data_path, file))
        if image.shape[2] > 3:
            image = image[:, :, :3]
            io.imsave(os.path.join(output_path, file), image)


def image_filter_pixels(image_path, return_image=False):
    """
    Find ratio of black (0) and white (255) pixels to total pixels in image.
    Also returns if image is rgb or not and whether image is in manga aspect
    ratio range or not.
    """
    if type(image_path) is WindowsPath or type(image_path) is str:
        image = io.imread(image_path)
    else:
        image = image_path
    rgb = False
    manga_ar = check_aspect_ratio(image)
    if len(image.shape) == 3:
        rgb = check_rgb(image)
        image = np.average(image, axis=2)
    white_ratio = np.count_nonzero(image == 255) / (image.shape[0] * image.shape[1])
    black_ratio = np.count_nonzero(image == 0) / (image.shape[0] * image.shape[1])
    if return_image:
        return white_ratio, black_ratio, rgb, manga_ar, image
    return white_ratio, black_ratio, rgb, manga_ar


def find_iqr(base_path):
    """
    Walk through all the images in provided path and find distribution of white
    and black pixel ratios.
    """
    filepaths = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                filepaths.append(os.path.join(root, file))
    white_ratios = []
    black_ratios = []
    for file in tqdm(filepaths, desc="Finding iqr"):
        white_ratio, black_ratio, rgb, manga_ar = image_filter_pixels(file)
        if manga_ar and not rgb:
            white_ratios.append(white_ratio)
            black_ratios.append(black_ratio)
    return white_ratios, black_ratios


def pdf_to_image(pdf_path, op_dir, image_base_name, zoom=4):
    """
    Function to convert pdf to images.
    """
    if type(op_dir) == str:
        op_dir = Path(op_dir)
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(zoom, zoom)
    pages = len(doc)
    for i in range(pages):
        fpath = op_dir / image_base_name.format(i)
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat)
        pix.save(fpath)
    doc.close()


def save_checkpoint(model, optimizer, filename):
    """
    Function to save checkpoint for GANs.
    """
    print("**SAVING CHECKPOINT**")
    checkpoint = {
        "model": model.state_dict(),  # saves the models current weights and biases
        "optimizer": optimizer.state_dict(),  # saves the optimizers state and hyperparameters
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer=None, lr=None):
    """
    Function to load checkpoint for GANs.
    """
    print("**LOADING CHECKPOINT**")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(
        checkpoint[list(checkpoint.keys())[0]]
    )  # checkpoint["model"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lr:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
