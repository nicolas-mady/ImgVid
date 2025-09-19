import numpy as np
from collections import Counter


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
    for row in img:
        for col in row:
            b, g, r = map(int, col)
            hist[r] += 1
            hist[g + 256] += 1
            hist[b + 512] += 1
    if ret:
        return hist
    with open("global-histogram.txt", "w") as f:
        f.write(str(hist))
    print("✅ Saved global-histogram.txt")


def local_hist(img, k=3, o='h'):
    d = {o: img.shape[0] // k}
    if o == 'h':
        h = img.shape[0] // k
        pts = [img[i*h:(i+1)*h] for i in range(k)]
    else:
        w = img.shape[1] // k
        pts = [img[:, i*w:(i+1)*w] for i in range(k)]
    hist = []
    for pt in pts:
        hist += global_hist(pt, ret=True)
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


def sawtooth_wave(img):
    return np.uint8(img % 64 / 63 * 255)


def logarithmic_transformation(img):
    c = 255 / np.log(1 + 255)
    return np.uint8(c * np.log(1 + img))


def mean_filter(img, w=7, it=3):
    new = np.zeros_like(img) + 255
    rows, cols, *_ = img.shape
    p = w // 2
    for i in range(p, rows - p):
        for j in range(p, cols - p):
            new[i, j] = np.mean(img[i-p:i+p+1, j-p:j+p+1], axis=(0, 1))
    new = np.uint8(new)
    if it == 1:
        return new
    return mean_filter(new, w, it - 1)


def k_neighbor(img, k=6, it=3):
    new = np.zeros_like(img) + 255
    rows, cols, *_ = img.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            window = img[i-1:i+2, j-1:j+2]
            if _:
                neighbors = window.reshape(-1, img.shape[2])
                neighbors = np.sort(neighbors, axis=0)[-k:]
                new[i, j] = np.mean(neighbors, axis=0)
            else:
                neighbors = np.sort(window.flatten())[-k:]
                new[i, j] = np.mean(neighbors)
    new = np.uint8(new)
    if it == 1:
        return new
    return k_neighbor(new, k, it - 1)


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
    new = np.zeros_like(img) + 255
    rows, cols, *_ = img.shape
    p = w // 2
    for i in range(p, rows - p):
        for j in range(p, cols - p):
            if _:
                new[i, j] = Counter(img[i-p:i+p+1, j-p:j+p+1].reshape(-1, img.shape[2]).tolist()).most_common(1)[0][0]
            else:
                new[i, j] = Counter(img[i-p:i+p+1, j-p:j+p+1].ravel()).most_common(1)[0][0]
    new = np.uint8(new)
    if it == 1:
        return new
    return moden_filter(new, w, it - 1)

