import torch
import os

# Device settings
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
KWARGS = {'pin_memory': True, 'num_workers': 4} if DEVICE == 'cude' else {}

# Input data paths
DATA_PATH = './data'
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train')
COLLECTED_PATH = os.path.join(DATA_PATH, 'collected')
METADATA_PATH = os.path.join(DATA_PATH, 'satellite_data.csv')
COORDINATES_PATH = os.path.join(DATA_PATH, 'coords.json')

# Data attributes
SPLIT_RATE = 0.1
BATCH_SIZE = 16
PATCH_SIZE = (256, 256)

# Learning parameters
LEARNING_RATE = 0.0005
EPOCHS = 50
DEEP_SUPERVISION = True

# Output data paths
OUTPUT_PATH = './output'
MODEL_PATH = os.path.join(OUTPUT_PATH, 'nestedunet_model.pth')
HISTORY_PATH = os.path.join(OUTPUT_PATH, 'nestedunet_history.pickle')
MASKS_PATH = os.path.join(OUTPUT_PATH, 'masks')
