import os
import numpy as np
import config
import re
from skimage import io
from skimage.filters import threshold_otsu
from utils.index_metrics import nbr, nbr2, bais2


def calculate_metrics(img_pre, img_post, mask):
    def dice_coefficient(img1, img2):
        n = len(img1)

        img1 = [img.reshape(-1) for img in img1]
        img2 = [img.reshape(-1) for img in img2]
        intersection = [(imgs[0] * imgs[1]).sum() for imgs in zip(img1, img2)]
        dice = [(2.0 * imgs[0]) / (imgs[1].sum() + imgs[2].sum()) for imgs in zip(intersection, img1, img2)]

        return 1 - sum(dice) / n

    def iou_score(img1, img2):
        n = len(img1)

        img1 = [img.reshape(-1) for img in img1]
        img2 = [img.reshape(-1) for img in img2]

        intersection = [(imgs[0] * imgs[1]).sum() for imgs in zip(img1, img2)]
        union = [(imgs[1] + imgs[2]).sum() - imgs[0] for imgs in zip(intersection, img1, img2)]
        iou = [pair[0] / pair[1] for pair in zip(intersection, union)]

        return sum(iou) / n

    def threshold_evaluate(diffs):
        thr = [threshold_otsu(diff) for diff in diffs]
        thr = [np.where(pair[0] > pair[1], 1, 0) for pair in zip(diffs, thr)]
        dice = dice_coefficient(thr, mask)
        iou = iou_score(thr, mask)
        return dice, iou

    dnbr = [nbr(pair[0]) - nbr(pair[1]) for pair in zip(img_pre, img_post)]
    nbr_dice, nbr_iou = threshold_evaluate(dnbr)

    dnbr2 = [nbr2(pair[0]) - nbr2(pair[1]) for pair in zip(img_pre, img_post)]
    nbr2_dice, nbr2_iou = threshold_evaluate(dnbr2)

    dbais2 = [bais2(pair[0]) - bais2(pair[1]) for pair in zip(img_pre, img_post)]
    bais2_dice, bais2_iou = threshold_evaluate(dbais2)

    return nbr_dice, nbr_iou, nbr2_dice, nbr2_iou, bais2_dice, bais2_iou


paths = sorted([
    os.path.join(config.TRAIN_DATA_PATH, name)
    for name in os.listdir(config.TRAIN_DATA_PATH)
    if os.path.isdir(os.path.join(config.TRAIN_DATA_PATH, name))
])

imgs_pre, imgs_post, masks = list(), list(), list()
for path in paths:
    files = sorted(
        [name for name in os.listdir(path) if re.match(".+[^coverage].tiff$", name)])

    mask = io.imread(os.path.join(path, files[0]))
    mask = np.where(mask > 32, 1, 0)

    masks.append(mask)

    s2_images = [name for name in files if name.startswith('sentinel2')]

    img_pre = io.imread(os.path.join(path, s2_images[0]))
    img_post = io.imread(os.path.join(path, s2_images[-1]))

    imgs_pre.append(img_pre)
    imgs_post.append(img_post)

nbr_dice, nbr_iou, nbr2_dice, nbr2_iou, bais2_dice, bais2_iou = calculate_metrics(imgs_pre, imgs_post, masks)
print(nbr_dice, nbr_iou, nbr2_dice, nbr2_iou, bais2_dice, bais2_iou)
