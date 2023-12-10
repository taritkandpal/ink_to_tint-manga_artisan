import os
import torch

PROJECT_PATH = r"D:\Projects\ink_to_tint-manga_artisan"
os.chdir(PROJECT_PATH)

# Generic configs
SEED = 22
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASPECT_RATIO_LB = 1.2
ASPECT_RATIO_UB = 1.8
RESIZED_IMAGE_2D_DIM = (342, 512)
PRETRAINED_MODELS = "pretrained_models"
RESULTS = "results"
SAVED_MODELS = os.path.join(RESULTS, "saved_models")

# GAN configs
MEAN = 0.5
STD = 0.5
NORM_MEAN = [MEAN, MEAN, MEAN]
NORM_STD = [STD, STD, STD]
INPUT_DIR = "Datasets/Colorization/Final/bw"
TARGET_DIR = "Datasets/Colorization/Final/coloured"
TRAIN_RATIO = None  # 0.999
NUM_SAMPLES_PER_EPOCH = 16
LEARNING_RATE = 1e-7  # 1e-6 8e-6 4e-5 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 8
IMAGE_PAD_SIZE = 512
IMAGE_SIZE = 512
L1_LAMBDA = 100
NUM_EPOCHS = 20
LOAD_MODEL = True
CHECKPOINT_DISC = os.path.join(SAVED_MODELS, "disc.pth.tar")
CHECKPOINT_GEN = os.path.join(SAVED_MODELS, "gen.pth.tar")
REAL_IMAGES = "Datasets/Colorization/Real"
TEST_FOLDER = os.path.join(RESULTS, "gan_images")

# Diffusion model configs
# whether to run diffusion in inference only mode or not
DIFFUSION_INFERENCE_ONLY = False
# A folder containing the training data. https://huggingface.co/docs/datasets/image_dataset#imagefolder
DIFFUSION_TRAIN_DATA = r"Datasets/StyleChange/ImageFolder/train"
# A folder containing the validation data.
DIFFUSION_VAL_DATA = r"Datasets/StyleChange/ImageFolder/test"
# A folder containing the complete data pool.
DIFFUSION_POOL_DATA = r"Datasets/StyleChange/ImageFolder/pool"
# Directory for diffusion model output (model files and images)
DIFFUSION_OUTPUT_DIR = os.path.join(SAVED_MODELS, "instruct-pix2pix-model")
# Path to pretrained model or model identifier from huggingface.co/models.
DIFFUSION_MODEL = DIFFUSION_OUTPUT_DIR  # "Meina/MeinaMix_V10"
# Revision of pretrained model identifier from huggingface.co/models.
DIFFUSION_MODEL_REVISION = None
# Imagefolder metadata column names
DIFFUSION_ORIGINAL_IMAGE_COL = "input_image"
DIFFUSION_EDITED_IMAGE_COL = "edited_image"
DIFFUSION_EDIT_PROMPT_COL = "edit_prompt"
# Directory to store validation output
DIFFUSION_VALIDATION_OUTPUT_DIR = os.path.join(
    DIFFUSION_OUTPUT_DIR, "validation_images"
)
# Mixed precision training for efficiency - ["no", "fp16", "bf16"]
DIFFUSION_MIXED_PRECISION = "fp16"
# Number of update steps (to accumulate gradients) before doing a backward pass
DIFFUSION_GRAD_ACC_STEPS = 1
# Resolution of images in filesystem for input
DIFFUSION_INPUT_SIZE = 600
# Resolution of images to be used for diffusion
DIFFUSION_IMAGE_SIZE = 512
# Validation data size
DIFFUSION_TEST_SIZE = 32
# Number of images to be generated during validation
DIFFUSION_NUM_VAL_IMAGES = DIFFUSION_TEST_SIZE
# Training batch size
DIFFUSION_BATCH_SIZE = 4
# Number of training epochs
DIFFUSION_TRAIN_EPOCHS = 1
# Number of epochs after which validation is to be used
DIFFUSION_VAL_EPOCHS = 1
# Cap on the number of datapoints to use from training out of total available
DIFFUSION_MAX_TRAIN_SAMPLES = 1440
# Initial Adam LR
DIFFUSION_ADAM_LR = 1e-4
# Scheduler to use - ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
DIFFUSION_LR_SCHEDULER = "cosine"
# Diffusion generation parameters
DIFFUSION_NUM_INFERENCE_STEPS = 150  # 100
DIFFUSION_IMAGE_GUIDANCE_SCALE = 30  # 1.5
DIFFUSION_GUIDANCE_SCALE = 6  # 7
