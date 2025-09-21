import numpy as np
import basics
import cv2 as cv


def bg_branco(img, k=64):

    def sobel(img):
        img = basics.grey(img)

        new = np.zeros_like(img, dtype=np.float32)

        mx = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
        my = mx.T

        h, w = img.shape

        for i in range(1, h-1):
            for j in range(1, w-1):
                janela = img[i-1:i+2, j-1:j+2]

                gx = np.sum(janela * mx)
                gy = np.sum(janela * my)

                new[i, j] = np.sqrt(gx**2 + gy**2)

        # return np.uint8(np.clip(new, 0, 255))
        new = cv.normalize(new, None, 0, 255, cv.NORM_MINMAX)
        return np.uint8(new)

    # img_quantizada = quantizacao_kmeans(img, k)
    bordas_sobel = sobel(img)
    img_bordas_branco = np.ones_like(img) * 255
    mask_sobel = bordas_sobel > 50
    img_bordas_branco[mask_sobel] = [255, 0, 0]
    return img_bordas_branco


def bg_original(img, k=64):

    def roberts(img):
        img = basics.grey(img)
        m1 = np.array([[1,0],[0,-1]], dtype=np.float32)
        m2 = np.array([[0,1],[-1,0]], dtype=np.float32)
        m, n = img.shape
        for i in range(m-1):
            for j in range(n-1):
                w = img[i:i+2, j:j+2]
                gx = (w * m1).sum()
                gy = (w * m2).sum()
                img[i, j] = np.sqrt(gx**2 + gy**2)
        return np.uint8(cv.normalize(img, None, 0, 255, cv.NORM_MINMAX))

    # img_quantizada = quantizacao_kmeans(img, k)
    bordas_sobel = roberts(img)
    img_bordas_branco = np.ones_like(img) * 255
    mask_sobel = bordas_sobel > 50
    img_bordas_branco[mask_sobel] = [255, 0, 0]
    img_bordas_original = img.copy()
    img_bordas_original[mask_sobel] = [255, 0, 0] 
    return img_bordas_original