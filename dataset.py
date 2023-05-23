import math
import os
import re
from random import shuffle
import albumentations as A
import numpy as np
from skimage import io
from torch import Tensor
from torch.utils.data import Dataset


class AerialDataset(Dataset):
    def __init__(self, paths: list, transforms: A.Compose, patch_size: tuple, train_set: bool,
                 split_rate: float) -> None:
        self.transforms = transforms
        self.patch_size = patch_size

        self.images, self.masks = list(), list()

        for path in paths:
            files = sorted(
                [os.path.join(path, name) for name in os.listdir(path) if re.match(".+[^coverage].tiff$", name)])
            images, masks = self._get_patches(io.imread(files[-1]), io.imread(files[0]))

            self.images.extend(images)
            self.masks.extend(masks)

        tmp = list(zip(self.images, self.masks))
        shuffle(tmp)
        self.images, self.masks = zip(*tmp)
        self.images, self.masks = list(self.images), list(self.masks)

        if train_set:
            self.images = self.images[int(split_rate * len(self.images)):]
            self.masks = self.masks[int(split_rate * len(self.masks)):]
        else:
            self.images = self.images[:int(split_rate * len(self.images))]
            self.masks = self.masks[:int(split_rate * len(self.masks))]

        self.images = np.stack(self.images)
        self.masks = np.stack(self.masks)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> (Tensor, Tensor):
        image = self.images[index]
        mask = self.masks[index]

        mask = np.where(mask > 32, 1, 0)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask

    def _get_patches(self, image: np.ndarray, mask: np.ndarray) -> (list, list):
        res = image.shape[:-1]
        max_i = math.ceil(res[0] / self.patch_size[0])
        max_j = math.ceil(res[1] / self.patch_size[1])

        images, masks = list(), list()
        for i in range(max_i):
            for j in range(max_j):
                vert = slice(min(i * self.patch_size[0], res[0] - self.patch_size[0]),
                             min((i + 1) * self.patch_size[0], res[0]))
                hor = slice(min(j * self.patch_size[1], res[1] - self.patch_size[1]),
                            min((j + 1) * self.patch_size[1], res[1]))

                images.append(image[vert, hor, :-1])
                masks.append(mask[vert, hor, np.newaxis])

        return images, masks
