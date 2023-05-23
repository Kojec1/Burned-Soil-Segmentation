import torch
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
KWARGS = {'pin_memory': True, 'num_workers': 4} if DEVICE == 'cude' else {}

TRAIN_DATA_PATH = './data'
COLLECTED_PATH = os.path.join(TRAIN_DATA_PATH, 'collected')
METADATA_PATH = os.path.join(TRAIN_DATA_PATH, 'satellite_data.csv')
COORDINATES_PATH = os.path.join(TRAIN_DATA_PATH, 'coords.json')

FOLDS = ['coral', 'cyan', 'grey', 'lime', 'magenta', 'pink', 'purple']

SPLIT_RATE = 0.1

CROP_HEIGHT = 2432
CROP_WIDTH = 2432

RESIZE_HEIGHT = 256
RESIZE_WIDTH = 256

BATCH_SIZE = 16

PATCH_SIZE = (256, 256)

LEARNING_RATE = 0.0005

EPOCHS = 50

DEEP_SUPERVISION = True

OUTPUT_PATH = './output'
MODEL_PATH = os.path.join(OUTPUT_PATH, 'nestedunet_model.pth')
HISTORY_PATH = os.path.join(OUTPUT_PATH, 'nestedunet_history.pickle')

