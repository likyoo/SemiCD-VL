import random
import math

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms


def crop(imgA, imgB, mask, size, ignore_value=255):
    w, h = imgA.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    imgA = ImageOps.expand(imgA, border=(0, 0, padw, padh), fill=0)
    imgB = ImageOps.expand(imgB, border=(0, 0, padw, padh), fill=0)
    if isinstance(mask, list):
        mask_1 = []
        for m in mask:
            mask_1.append(ImageOps.expand(m, border=(0, 0, padw, padh), fill=ignore_value))
    else:
        mask_1 = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = imgA.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    imgA = imgA.crop((x, y, x + size, y + size))
    imgB = imgB.crop((x, y, x + size, y + size))
    if isinstance(mask_1, list):
        mask_2 = []
        for m in mask_1:
            mask_2.append(m.crop((x, y, x + size, y + size)))
    else:
        mask_2 = mask_1.crop((x, y, x + size, y + size))

    return imgA, imgB, mask_2


def hflip(imgA, imgB, mask, p=0.5):
    if random.random() < p:
        imgA = imgA.transpose(Image.FLIP_LEFT_RIGHT)
        imgB = imgB.transpose(Image.FLIP_LEFT_RIGHT)
        if isinstance(mask, list):
            mask_ = []
            for m in mask:
                mask_.append(m.transpose(Image.FLIP_LEFT_RIGHT))
        else:
            mask_ = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return imgA, imgB, mask_
    return imgA, imgB, mask


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        if isinstance(mask, list):
            mask_ = []
            for m in mask:
                mask_.append(torch.from_numpy(np.array(m)).long())
        else:
            mask_ = torch.from_numpy(np.array(mask)).long()
        return img, mask_
    return img


def resize(imgA, imgB, mask, ratio_range):
    w, h = imgA.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    imgA = imgA.resize((ow, oh), Image.BILINEAR)
    imgB = imgB.resize((ow, oh), Image.BILINEAR)
    if isinstance(mask, list):
        mask_ = []
        for m in mask:
            mask_.append(m.resize((ow, oh), Image.NEAREST))
    else:
        mask_ = mask.resize((ow, oh), Image.NEAREST)
    return imgA, imgB, mask_


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask
