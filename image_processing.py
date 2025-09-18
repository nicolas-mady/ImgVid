import numpy as np


def bright(img):
    try:
        factor = float(input())
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


def lin_contrast(img):
    a = np.min(img)
    b = np.max(img)
    return np.uint8((img - a) * 255 / (b - a))


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


def thresholding(img, th=128, n=255):
    return np.uint8((img >= th) * n)

