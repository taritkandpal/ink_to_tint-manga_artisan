from pathlib import Path
import os
import random

from skimage import io
from tqdm import tqdm

from utils import ldict_to_jsonl
from config import *


def find_orig_from_pool(original_pool, edited_class):
    # random.shuffle(original_pool)
    plen = len(original_pool)
    while plen:
        plen -= 1
        randidx = random.randrange(len(original_pool))
        fname, original_class = original_pool[randidx]
        if original_class != edited_class:
            del original_pool[randidx]
            return fname, original_class
    return "Failed", "Failed"


classes = [
    "MobPsycho",
    "Naruto",
    "OnePiece",
    "Jojo",
]
ip_folder = Path("Datasets/StyleChange/Final")
training_op_folder = Path("Datasets/StyleChange/ImageFolder")

edited_list = {}
original_pool = []
for data_class in classes:
    data_path = ip_folder / data_class
    fnames = os.listdir(data_path)
    edited_list[data_class] = fnames
    original_pool.extend([(fname, data_class) for fname in fnames])
random.shuffle(original_pool)

ldict = []
failed = []
for edited_class in classes:
    fnames = edited_list[edited_class]
    for fname in tqdm(fnames, desc=f"{edited_class}"):
        data = dict()
        data["edited_image"] = fname
        data["edited_class"] = edited_class
        data["edit_prompt"] = (
            f"Convert the given manga to {edited_class} art style without altering "
            f"the story and the characters of the input."
        )
        original_fname, original_class = find_orig_from_pool(original_pool, edited_class)
        if original_fname != "Failed" and original_class != "Failed":
            data["input_image"] = original_fname
            data["input_class"] = original_class
            ldict.append(data)
        else:
            failed.append(data)
random.shuffle(ldict)
print(len(ldict))
print(len(failed))
