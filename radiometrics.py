import numpy as np


def contraste_linear(img, a=0, b=255):
    new = img.copy()
    for i, row in enumerate(img):
        for j, px in enumerate(row):
            if px < a:
                new[i, j] = 0
            elif px > b:
                new[i, j] = 255
            else:
                new[i, j] = (px - a) / (b - a) * 255
    return new


def compressao_expansao(img):
    new = img.copy()
    for i, row in enumerate(img):
        for j, px in enumerate(row):
            if px <= 85:
                new[i, j] = img[i, j] // 2
            elif px < 170:
                new[i, j] = img[i, j] * 2 - 127
            else:
                new[i, j] = img[i, j] // 2 + 128
    return np.uint8(new)


def dente_serra(img):
    return np.uint8(img % 64 / 63 * 255)


def log(img):
    # c = 255 / np.log(1 + 255)
    return np.uint8(46 * np.log(np.clip(img, 1, 255)))


def threshold(img, *, th=128, n=255, wb=False):
    img = img < th if wb else img >= th
    return np.uint8(np.clip(img * n, 0, 255))