import numpy as np


def bright(img):
    try:
        factor = float(input("fator de brilho:\n"))
    except ValueError:
        factor = 1.25
    return np.uint8(np.clip(img * factor, 0, 255))


def grey(img):
    return np.uint8(img @ [.1, .6, .3])


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


def linear_contrast_expansion(img, a=0, b=255):
    new = np.zeros_like(img)
    for i, row in enumerate(img):
        for j, px in enumerate(row):
            if px < a:
                new[i, j] = 0
            elif px > b:
                new[i, j] = 255
            else:
                new[i, j] = (px - a) / (b - a) * 255
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
    c = 255 / np.log(1 + 255)
    return np.uint8(c * np.log(1 + img))


def mean_filter(img, w=7, it=3):
    new = np.zeros_like(img)
    rows, cols, *_ = img.shape
    p = w // 2
    for i in range(p, rows - p):
        for j in range(p, cols - p):
            new[i, j] = np.mean(img[i-p:i+p+1, j-p:j+p+1], axis=(0, 1))
    new = np.uint8(new)
    if it == 1:
        return new
    return mean_filter(new, w, it - 1)


def k_neighbor(img, it=3):
    new = np.zeros_like(img)
    rows, cols, *_ = img.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighbors = [
                img[i-1, j-1], img[i-1, j], img[i-1, j+1],
                img[i, j], img[i, j+1],
                img[i+1, j], img[i+1, j+1]
            ]
            if _:
                new[i, j] = np.mean(neighbors, axis=0)
            else:
                new[i, j] = np.mean(neighbors)
    new = np.uint8(new)
    if it == 1:
        return new
    return k_neighbor(new, it - 1)


def median_filter(img, w=7, it=3):
    new = np.zeros_like(img)
    rows, cols, *_ = img.shape
    p = w // 2
    for i in range(p, rows - p):
        for j in range(p, cols - p):
            new[i, j] = np.median(img[i-p:i+p+1, j-p:j+p+1], axis=(0, 1))
    new = np.uint8(new)
    if it == 1:
        return new
    return median_filter(new, w, it - 1)


def moden_filter(img, w=7, it=3):
    new = np.zeros_like(img)
    rows, cols, *_ = img.shape
    p = w // 2
    for i in range(p, rows - p):
        for j in range(p, cols - p):
            new[i, j] = np.bincount(img[i-p:i+p+1, j-p:j+p+1].ravel(), minlength=256).argmax()
    new = np.uint8(new)
    if it == 1:
        return new
    return moden_filter(new, w, it - 1)