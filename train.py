import os
import pickle
import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.optim import Adam
from torch.utils.data import DataLoader
import config
from dataset import AerialDataset
from loss import BCEDiceLoss, dice_coef, iou_score, accuracy_score
from model import NestedUNet


# Creat a list of paths
paths = sorted([
    os.path.join(config.TRAIN_DATA_PATH, name)
    for name in os.listdir(config.TRAIN_DATA_PATH)
    if os.path.isdir(os.path.join(config.TRAIN_DATA_PATH, name))
])

# Create transformation compositions
transform_train = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    ToTensorV2(transpose_mask=True)
])
transform_test = A.Compose([
    ToTensorV2(transpose_mask=True)
])

# Load the train and test datasets
train_set = AerialDataset(paths, transform_train, config.PATCH_SIZE, train_set=True, split_rate=config.SPLIT_RATE)
test_set = AerialDataset(paths, transform_test, config.PATCH_SIZE, train_set=False, split_rate=config.SPLIT_RATE)

# Initiate the train and test loaders
train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, **config.KWARGS)
test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, **config.KWARGS)

# Create the model
model = NestedUNet(1, config.PATCH_SIZE, config.DEEP_SUPERVISION).to(config.DEVICE)

# Initiate optimizer and loss function
optim = Adam(model.parameters(), lr=config.LEARNING_RATE)
criterion = BCEDiceLoss()

# Get the number of train and test steps
train_steps = len(train_set) // config.BATCH_SIZE
test_steps = len(test_set) // config.BATCH_SIZE

# Initiate a train history
history = {'train_loss': [], 'train_dice': [], 'train_iou': [], 'train_acc': [],
           'test_loss': [], 'test_dice': [], 'test_iou': [], 'test_acc': []}

# Train the model
for epoch in range(config.EPOCHS):
    print('EPOCH: {}'.format(epoch))
    # Set the model in learning mode
    model.train()

    # Initiate train and test loss, dice, iou and acc values
    train_loss, train_dice, train_iou, train_acc = 0., 0., 0., 0.
    test_loss, test_dice, test_iou, test_acc = 0., 0., 0., 0.

    # Loop through the training data
    for i, (images, targets) in enumerate(train_loader):
        # Move the data to the device
        images, targets = images.to(config.DEVICE), targets.to(config.DEVICE)

        # Forward pass
        outputs = model(images)

        loss = 0.
        if config.DEEP_SUPERVISION:  # Deep Supervision
            for output in outputs:
                loss += criterion(output, targets)
            loss /= len(outputs)
            outputs = outputs[-1]
        else:  # Single output
            loss = criterion(outputs, targets)

        # Backward step
        optim.zero_grad()
        loss.backward()
        # Update weights
        optim.step()

        # Count the metric values
        dice = dice_coef(outputs, targets)
        iou = iou_score(outputs, targets)
        acc = accuracy_score(outputs, targets)

        # Update the loss and metrics values for the training
        train_loss += loss.detach().item()
        train_dice += dice.detach().item()
        train_iou += iou.detach().item()
        train_acc += acc.detach().item()

        if i % 10 == 0:
            print('\tIteration: {} Loss: {:.2f} Dice: {:.2f} IoU: {:.2f} Acc: {:.2f}'.format(i, loss, dice, iou, acc))

    # Evaluate the model
    with torch.no_grad():
        # Set the model in evaluation mode
        model.eval()
        # Loop through the test data
        for (images, targets) in test_loader:
            # Move the data to the device
            images, targets = images.to(config.DEVICE), targets.to(config.DEVICE)

            # Predict values
            outputs = model(images)

            loss = 0.
            if config.DEEP_SUPERVISION:  # Deep Supervision
                for output in outputs:
                    loss += criterion(output, targets)
                loss /= len(outputs)
                outputs = outputs[-1]
            else:  # Single output
                loss = criterion(outputs, targets)

            # Count the metric values
            dice = dice_coef(outputs, targets)
            iou = iou_score(outputs, targets)
            acc = accuracy_score(outputs, targets)

            # Update the loss and metrics values for the testing
            test_loss += loss.item()
            test_dice += dice.item()
            test_iou += iou.item()
            test_acc += acc.item()

    # Calculate the average loss, dice, iou and acc for the training
    avg_train_loss = train_loss / train_steps
    history['train_loss'].append(avg_train_loss)
    avg_train_dice = train_dice / train_steps
    history['train_dice'].append(avg_train_dice)
    avg_train_iou = train_iou / train_steps
    history['train_iou'].append(avg_train_iou)
    avg_train_acc = train_acc / train_steps
    history['train_acc'].append(avg_train_acc)

    # Calculate the average loss, dice, iou and acc for the testing
    avg_test_loss = test_loss / test_steps
    history['test_loss'].append(avg_test_loss)
    avg_test_dice = test_dice / test_steps
    history['test_dice'].append(avg_test_dice)
    avg_test_iou = test_iou / test_steps
    history['test_iou'].append(avg_test_iou)
    avg_test_acc = test_acc / test_steps
    history['test_acc'].append(avg_test_acc)

    print('Train: Loss {:.2f} Dice {:.2f} IoU {:.2f} Acc {:.2f} | Test: Loss {:.2f} Dice {:.2f} IoU {:.2f} Acc {:.2f}'
          .format(avg_train_loss, avg_train_dice, avg_train_iou, avg_train_acc,
                  avg_test_loss, avg_test_dice, avg_test_iou, avg_test_acc))

# Save the model history dict
torch.save(model, config.MODEL_PATH)
with open(config.HISTORY_PATH, 'wb') as f:
    pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)
