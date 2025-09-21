import numpy as np
import cv2 as cv
from basics import global_hist


def bic(img, k=64):
    dados = img.reshape((-1, 3))
    dados = np.float32(dados)

    criterios = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, rotulos, centros = cv.kmeans(dados, k, None, criterios, 10, cv.KMEANS_RANDOM_CENTERS)
    centros = np.uint8(centros)

    img_quant = centros[rotulos.flatten()]
    img_quant = img_quant.reshape(img.shape)

    img_bordas = np.ones_like(img) * 255
    img_interior = np.ones_like(img) * 255

    m, n = img.shape[:2]

    for i in range(1, m-1):
        for j in range(1, n-1):
            px = img_quant[i, j]
            if (np.array_equal(px, img_quant[i-1, j])
                and np.array_equal(px, img_quant[i, j+1])
                and np.array_equal(px, img_quant[i+1, j])
                and np.array_equal(px, img_quant[i, j-1])
            ):
                img_interior[i, j] = img[i, j]
            else:
                img_bordas[i, j] = img[i, j]

    global_hist(img_bordas, 'q8.1.txt')
    global_hist(img_interior, 'q8.2.txt')
    return img_interior, img_bordas