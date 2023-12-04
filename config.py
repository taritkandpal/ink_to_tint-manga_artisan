import os
import torch

PROJECT_PATH = r"D:\Projects\ink_to_tint-manga_artisan"
os.chdir(PROJECT_PATH)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 22

ASPECT_RATIO_LB = 1.2
ASPECT_RATIO_UB = 1.8

RESIZED_IMAGE_2D_DIM = (342, 512)  # (400, 600) (170, 256)
RESIZED_IMAGE_3D_DIM = (400, 600, 3)

RESIZED_IMAGE_2D_DIM_PAD = (400, 600)
RESIZED_IMAGE_3D_DIM_PAD = (400, 600, 3)

PRETRAINED_MODELS = "pretrained_models"
RESULTS = "results"
SAVED_MODELS = os.path.join(RESULTS, "saved_models")

# GAN configs
INPUT_DIR = "Datasets/Colorization/Final/bw"
TARGET_DIR = "Datasets/Colorization/Final/coloured"
TRAIN_RATIO = None  # 0.999
NUM_SAMPLES_PER_EPOCH = 16
LEARNING_RATE = 4e-5
MIN_LR = 5e-5
BATCH_SIZE = 16
NUM_WORKERS = 8
IMAGE_PAD_SIZE = 512
IMAGE_SIZE = 512
CHANNEL_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 20
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_DISC = os.path.join(SAVED_MODELS, "disc.pth.tar")
CHECKPOINT_GEN = os.path.join(SAVED_MODELS, "gen.pth.tar")
REAL_IMAGES = "Datasets/Colorization/Real"
TEST_FOLDER = os.path.join(RESULTS, "gan_images")

MEAN = 0.5
STD = 0.5
NORM_MEAN = [MEAN, MEAN, MEAN]
NORM_STD = [STD, STD, STD]

# Diffusion model configs
DIFFUSION_MODEL = r"meinamix_meinaV10.safetensors"
DIFFUSION_TRAIN_DATA = r"Datasets/StyleChange/Debug"
DIFFUSION_VAL_DATA = r"Datasets/StyleChange/Debug"
DIFFUSION_OUTPUT_DIR = os.path.join(SAVED_MODELS, "instruct-pix2pix-model")
DIFFUSION_MIXED_PRECISION = "fp16"
DIFFUSION_NUM_VAL_IMAGES = 8
DIFFUSION_IMAGE_SHAPE = 512
DIFFUSION_BATCH_SIZE = 2
DIFFUSION_TRAIN_EPOCHS = 1
DIFFUSION_NUM_INFERENCE_STEPS = 5
DIFFUSION_ADAM_LR = 1e-4
DIFFUSION_ADAM_BETA1 = 0.9
DIFFUSION_ADAM_BETA2 = 0.999
DIFFUSION_ADAM_DECAY = 0.01
DIFFUSION_LR_SCHEDULER = "cosine"
