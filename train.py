import os
import pickle

import albumentations as A
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.optim import Adam
from torch.utils.data import DataLoader

import config
from dataset import AerialDataset
from loss import BCEDiceLoss, dice_coef, iou_score
from model import NestedUNet

# metadata = pd.read_csv(config.METADATA_PATH)
#
# paths = sorted([
#     os.path.join(config.TRAIN_DATA_PATH, name)
#     for name in metadata[metadata.fold == config.FOLDS[0]].folder.tolist()
# ])

paths = sorted([
    os.path.join(config.TRAIN_DATA_PATH, name)
    for name in os.listdir(config.TRAIN_DATA_PATH)
    if os.path.isdir(os.path.join(config.TRAIN_DATA_PATH, name))
])

# mask_paths = sorted([
#     os.path.join(config.TRAIN_DATA_PATH, name)
#     for name in os.listdir(config.TRAIN_DATA_PATH)
#     if name.endswith('.png')
# ])

# tmp = list(zip(img_paths, mask_paths))
# tmp = list(filter(lambda x: np.any(np.all(cv2.imread(x[1]) == np.array([255, 255, 255]), axis=-1)), tmp))
# shuffle(tmp)
# img_paths, mask_paths = zip(*tmp)
# img_paths, mask_paths = list(img_paths), list(mask_paths)


# train_imgs = img_paths[int(config.SPLIT_RATE * len(img_paths)):]
# train_masks = mask_paths[int(config.SPLIT_RATE * len(mask_paths)):]
# test_imgs = img_paths[:int(config.SPLIT_RATE * len(img_paths))]
# test_masks = mask_paths[:int(config.SPLIT_RATE * len(mask_paths))]

transform_train = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    ToTensorV2(transpose_mask=True)
])

transform_test = A.Compose([
    ToTensorV2(transpose_mask=True)
])

# paths = ['data/EMSR207_01MIRANDADOCORVO_02GRADING_MAP_v2_vector']

train_set = AerialDataset(paths, transform_train, config.PATCH_SIZE, train_set=True, split_rate=config.SPLIT_RATE)
test_set = AerialDataset(paths, transform_test, config.PATCH_SIZE, train_set=False, split_rate=config.SPLIT_RATE)

train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, **config.KWARGS)
test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, **config.KWARGS)

model = NestedUNet(1, config.PATCH_SIZE, config.DEEP_SUPERVISION).to(config.DEVICE)

optim = Adam(model.parameters(), lr=config.LEARNING_RATE)
criterion = BCEDiceLoss()

train_steps = len(train_set) // config.BATCH_SIZE
test_steps = len(test_set) // config.BATCH_SIZE

history = {'train_loss': [], 'train_dice': [], 'train_iou': [], 'test_loss': [], 'test_dice': [], 'test_iou': []}

for epoch in range(config.EPOCHS):
    print('EPOCH: {}'.format(epoch))
    model.train()

    train_loss, train_dice, train_iou = 0., 0., 0.
    test_loss, test_dice, test_iou = 0., 0., 0.

    for i, (images, targets) in enumerate(train_loader):
        images, targets = images.to(config.DEVICE), targets.to(config.DEVICE)
        outputs = model(images)

        loss = 0.
        if config.DEEP_SUPERVISION:
            for output in outputs:
                loss += criterion(output, targets)
            loss /= len(outputs)
        else:
            loss = criterion(outputs, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        dice = dice_coef(outputs[-1], targets)
        iou = iou_score(outputs[-1], targets)

        train_loss += loss.detach().item()
        train_dice += dice.detach().item()
        train_iou += iou.detach().item()

        if i % 10 == 0:
            print('\tIteration: {} Loss: {:.2f} Dice: {:.2f} IoU: {:.2f}'.format(i, loss, dice, iou))

    with torch.no_grad():
        # Set the model in evaluation mode
        model.eval()
        # Loop through the test data
        for (images, targets) in test_loader:
            images, targets = images.to(config.DEVICE), targets.to(config.DEVICE)
            outputs = model(images)

            loss = 0.
            if config.DEEP_SUPERVISION:
                for output in outputs:
                    loss += criterion(output, targets)
                loss /= len(outputs)
            else:
                loss = criterion(outputs, targets)

            dice = dice_coef(outputs[-1], targets)
            iou = iou_score(outputs[-1], targets)

            test_loss += loss
            test_dice += dice
            test_iou += iou

    avg_train_loss = train_loss / train_steps
    history['train_loss'].append(avg_train_loss)
    avg_train_dice = train_dice / train_steps
    history['train_dice'].append(avg_train_dice)
    avg_train_iou = train_iou / train_steps
    history['train_iou'].append(avg_train_iou)
    avg_test_loss = test_loss / test_steps
    history['test_loss'].append(avg_test_loss)
    avg_test_dice = test_dice / test_steps
    history['test_dice'].append(avg_test_dice)
    avg_test_iou = test_iou / test_steps
    history['test_iou'].append(avg_test_iou)

    print('Train Loss: {:.2f} Train Dice: {:.2f} Train IoU: {:.2f} Test Loss: {:.2f} Test Dice: {:.2f} Test IoU: {:.2f}'
          .format(avg_train_loss, avg_train_dice, avg_train_iou, avg_test_loss, avg_test_dice, avg_test_iou))

torch.save(model, config.MODEL_PATH)
with open(config.HISTORY_PATH, 'wb') as f:
    pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)
