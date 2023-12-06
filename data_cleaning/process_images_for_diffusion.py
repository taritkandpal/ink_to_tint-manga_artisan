import os
from pathlib import Path
import random

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
            f"Convert the given manga page to {edited_class} art style without altering "
            f"the story and the text."
        )
        original_fname, original_class = find_orig_from_pool(
            original_pool, edited_class
        )
        if original_fname != "Failed" and original_class != "Failed":
            data["input_image"] = original_fname
            data["input_class"] = original_class
            data["file_name"] = original_fname
            ldict.append(data)
        else:
            failed.append(data)

random.shuffle(ldict)
train_size = len(ldict) - DIFFUSION_TEST_SIZE
train = ldict[:train_size]
test = ldict[train_size:]
ldict_to_jsonl(train, training_op_folder / "train" / "metadata.jsonl")
ldict_to_jsonl(test, training_op_folder / "test" / "metadata.jsonl")
print("Done")
