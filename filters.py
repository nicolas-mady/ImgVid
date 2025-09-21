import numpy as np
from collections import Counter


def media(img, w=7, it=3):
    new = img.copy()
    rows, cols, *_ = img.shape
    p = w // 2
    for i in range(p, rows - p):
        for j in range(p, cols - p):
            new[i, j] = np.mean(img[i-p:i+p+1, j-p:j+p+1], axis=(0, 1))
    new = np.uint8(new)
    if it == 1:
        return new
    return media(new, w, it - 1)


def kvizinhos(img, k=6, it=3):
    new = img.copy()
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
    return kvizinhos(new, k, it - 1)


def mediana(img, w=7, it=3):
    new = np.zeros_like(img)
    rows, cols, *_ = img.shape
    p = w // 2
    for i in range(p, rows - p):
        for j in range(p, cols - p):
            new[i, j] = np.median(img[i-p:i+p+1, j-p:j+p+1], axis=(0, 1))
    new = np.uint8(new)
    if it == 1:
        return new
    return mediana(new, w, it - 1)


def moda(img, w=7, it=3):
    new = img.copy()
    rows, cols, *_ = img.shape
    p = w // 2
    for i in range(p, rows - p):
        for j in range(p, cols - p):
            if _:
                for k in range(3):
                    new[i, j, k] = Counter(img[i-p:i+p+1, j-p:j+p+1, k].ravel()).most_common(1)[0][0]
            else:
                new[i, j] = Counter(img[i-p:i+p+1, j-p:j+p+1].ravel()).most_common(1)[0][0]
    new = np.uint8(new)
    if it == 1:
        return new
    return moda(new, w, it - 1)


def sal_pimenta(img):
    new = img.copy()
    for row in new:
        for j in range(len(row)):
            if np.random.rand() <= 0.1:
                row[j] = 0 if np.random.rand() <= 0.5 else 255
    return new

