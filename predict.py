import torch
from torch import Tensor
import config
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import io
import matplotlib.pyplot as plt
from utils import image

img = io.imread('data/collected/roi_Canberra3_post.tif')

# TODO: TEMP SOLUTION!!!
# img = img[6000:, 6000:, :]

img = img / 10000.
print(img.shape)

res = img.shape[:-1]
max_i = res[0] // config.PATCH_SIZE[0]
max_j = res[1] // config.PATCH_SIZE[1]

transform = A.Compose([
    A.RandomCrop(max_i * config.PATCH_SIZE[0],  max_j * config.PATCH_SIZE[1]),
    ToTensorV2()
])

transformed = transform(image=img)
img = transformed['image']
print(img.shape)

images = list()
for i in range(max_i):
    for j in range(max_j):
        vert = slice(i * config.PATCH_SIZE[0], (i + 1) * config.PATCH_SIZE[0])
        hor = slice(j * config.PATCH_SIZE[1], (j + 1) * config.PATCH_SIZE[1])

        images.append(img[:, vert, hor].type(torch.float32))

images = torch.stack(images)
images = images.to(config.DEVICE)
print(images.shape)

n_images = images.shape[0]
n_steps = n_images // config.BATCH_SIZE

model = torch.load(config.MODEL_PATH)

outputs = list()
with torch.no_grad():
    for i in range(n_steps + 1):
        batch = images[i * config.BATCH_SIZE:min((i + 1) * config.BATCH_SIZE, n_images)]

        output = model(batch)
        output = torch.sigmoid(output[-1]).cpu().numpy()

        outputs.append(output)

outputs = np.concatenate(outputs)
print(outputs.shape)

plt.imshow(images[120, [11, 7, 3], :, :].cpu().permute(1, 2, 0))
plt.show()
plt.imshow(np.where(outputs[120][0] > 0.5, 1, 0), cmap='gray')
plt.show()
plt.imshow(images[220, [11, 7, 3], :, :].cpu().permute(1, 2, 0))
plt.show()
plt.imshow(np.where(outputs[220][0] > 0.5, 1, 0), cmap='gray')
plt.show()