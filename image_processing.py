import numpy as np
from math import log


def bright(img):
    try:
        factor = float(input("fator de brilho:\n"))
    except ValueError:
        factor = 1.2
    return np.uint8(np.clip(img * factor, 0, 255))


def grey(img):
    new = np.uint8(img @ [.1, .6, .3])
    print(new.shape)
    return new


def neg(img):
    return 255 - img


def global_hist(img, *, ret=False):
    hist = [0] * 768
    for ln in img:
        for col in ln:
            b, g, r = map(int, col)
            hist[r] += 1
            hist[g + 256] += 1
            hist[b + 512] += 1
    if ret:
        return hist
    with open("global-histogram.txt", "w") as f:
        f.write(str(hist))
    print("✅ Saved global-histogram.txt")


def local_hist(img):
    h = img.shape[0] // 3
    pt1 = img[:h]
    pt2 = img[h:2*h]
    pt3 = img[2*h:]
    hist = global_hist(pt1, 1) + global_hist(pt2, 1) + global_hist(pt3, 1)
    with open("local-histogram.txt", "w") as f:
        f.write(str(hist))
    print("✅ Saved local-histogram.txt")


def linear_contrast_expansion(img, i=0, a=0, b=255, n=255):
    new = np.zeros_like(img)
    for i, row in enumerate(img):
        for j, px in enumerate(row):
            if px < a:
                new[i, j] = 0
            elif px > b:
                new[i, j] = n
            else:
                new[i, j] = (px - a) / (b - a) * n
    return new

""" 
def thresholding(img, th=128, n=255):
    return np.uint8((img >= th) * n)


def histogram_equalization(img):
    hist = global_hist(img, ret=True)
    cdf = np.cumsum(hist)
    cdf_min = min(i for i in cdf if i > 0)
    total_pixels = img.shape[0] * img.shape[1]
    lut = np.uint8((cdf - cdf_min) / (total_pixels - cdf_min) * 255)

    new = np.zeros_like(img)
    for i, row in enumerate(img):
        for j, px in enumerate(row):
            new[i, j] = lut[px]
    return new """

def compression_and_expansion(img):
    new = np.zeros_like(img)
    for i, row in enumerate(img):
        for j, px in enumerate(row):
            if px <= 85:
                new[i, j] = img[i, j] // 2
            elif px < 170:
                new[i, j] = img[i, j] * 2 - 127
            else:
                new[i, j] = img[i, j] // 2 + 128
    return np.uint8(new)

# dente de serra
def sawtooth_wave(img):
    return np.uint8(img % 64 / 63 * 255)


def logarithmic_transformation(img):
    c = 255 / log(1 + 255)
    return np.uint8(c * np.log(1 + img))


