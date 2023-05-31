import os
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from skimage import io
import config


# Create a list of collected image paths
paths = sorted([
    os.path.join(config.COLLECTED_PATH, name)
    for name in os.listdir(config.COLLECTED_PATH)
    if name.endswith('post.tif')
])

# Iterate over paths
for path in paths:
    # Get a region of the current image
    region = path.split('_')[1]

    # Load the image
    img = io.imread(path)
    img = img / 10000.

    # Get the image resolution
    res = img.shape[:-1]
    max_i = res[0] // config.PATCH_SIZE[0]
    max_j = res[1] // config.PATCH_SIZE[1]

    # Crop the image so that the resolution is a multiple of the patch size
    transform = A.Compose([
        A.CenterCrop(max_i * config.PATCH_SIZE[0], max_j * config.PATCH_SIZE[1]),
        ToTensorV2()
    ])

    # Apply the transformation
    transformed = transform(image=img)
    img = transformed['image']

    # Split the image into patches
    images = list()
    for i in range(max_i):
        for j in range(max_j):
            vert = slice(i * config.PATCH_SIZE[0], (i + 1) * config.PATCH_SIZE[0])
            hor = slice(j * config.PATCH_SIZE[1], (j + 1) * config.PATCH_SIZE[1])

            images.append(img[:, vert, hor].type(torch.float32))

    images = torch.stack(images)
    images = images.to(config.DEVICE)

    # Get the number of steps
    n_images = images.shape[0]
    n_steps = n_images // config.BATCH_SIZE

    # Load the trained model
    model = torch.load(config.MODEL_PATH)

    # Predict masks for each patch
    outputs = list()
    # Turn off gradient calculation
    with torch.no_grad():
        for i in range(n_steps + 1):
            # Get a batch of images
            batch = images[i * config.BATCH_SIZE:min((i + 1) * config.BATCH_SIZE, n_images)]

            # Use the model to predict the output
            output = model(batch)
            output = torch.sigmoid(output[-1]).cpu().numpy()

            # Append output to list
            outputs.append(output)

    # Concatenate outputs from all batches
    outputs = np.concatenate(outputs)

    # Merge all the patches
    index = 0
    # Initiate mask array
    mask = np.zeros((max_i * config.PATCH_SIZE[0], max_j * config.PATCH_SIZE[1]))
    for i in range(max_i):
        for j in range(max_j):
            vert = slice(i * config.PATCH_SIZE[0], (i + 1) * config.PATCH_SIZE[0])
            hor = slice(j * config.PATCH_SIZE[1], (j + 1) * config.PATCH_SIZE[1])

            # Assign predicted values to corresponding pixels in mask array
            mask[vert, hor] = outputs[index, 0]
            index += 1

    # Threshold mask values to 0 or 1
    mask = np.where(mask > 0.5, 1, 0)

    # Save mask as image
    plt.imsave(os.path.join(config.MASKS_PATH, region + '_mask.png'), mask, cmap='gray')
