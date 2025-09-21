import numpy as np


def contraste_linear(img, a=0, b=255):
    new = img.copy()
    for i, row in enumerate(img):
        for j, px in enumerate(row):
            for k in range(3):
                if px[k] < a:
                    new[i, j, k] = 0
                elif px[k] > b:
                    new[i, j, k] = 255
                else:
                    new[i, j, k] = (px[k] - a) / (b - a) * 255
    return new


def compressao_expansao(img):
    new = img.copy()
    for i, row in enumerate(img):
        for j, px in enumerate(row):
            for k in range(3):
                if px[k] <= 85:
                    new[i, j, k] = img[i, j, k] // 2
                elif px[k] < 170:
                    new[i, j, k] = int(img[i, j, k]) * 2 - 127
                else:
                    new[i, j, k] = img[i, j, k] // 2 + 128
    return np.uint8(new)


def dente_serra(img):
    return np.uint8(img % 64 / 63 * 255)


def log(img):
    # c = 255 / np.log(1 + 255)
    return np.uint8(46 * np.log(np.clip(img, 1, 255)))