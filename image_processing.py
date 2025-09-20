import numpy as np
from collections import Counter
# import cv2 as cv


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
    return new
"""


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
    new = img.copy()
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


def mode_filter(img, w=7, it=3):
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
    return mode_filter(new, w, it - 1)


def pepper_and_salt(img):
    new = img.copy()
    for row in new:
        for j in range(len(row)):
            if np.random.rand() <= 0.1:
                row[j] = 0 if np.random.rand() <= 0.5 else 255
    return new


# def quantizacao_kmeans(img, k=64):
#     dados = img.reshape((-1, 3))
#     dados = np.float32(dados)
    
#     criterios = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     _, rotulos, centros = cv.kmeans(dados, k, None, criterios, 10, cv.KMEANS_RANDOM_CENTERS)
    
#     centros = np.uint8(centros)
#     img_quantizada = centros[rotulos.flatten()]
#     img_quantizada = img_quantizada.reshape(img.shape)
    
#     return img_quantizada


# def sobel(img):
#     if len(img.shape) == 3:
#         img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
#     new = np.zeros_like(img, dtype=np.float32)

#     mx = np.array([[-1, 0, 1], 
#                    [-2, 0, 2], 
#                    [-1, 0, 1]], dtype=np.float32)
    
#     my = np.array([[-1, -2, -1], 
#                    [ 0,  0,  0], 
#                    [ 1,  2,  1]], dtype=np.float32)
    
#     h, w = img.shape
    
#     for ln in range(1, h-1):
#         for col in range(1, w-1):
#             window = np.array([
#                 [img[ln-1, col-1], img[ln-1, col], img[ln-1, col+1]],
#                 [img[ln, col-1],   img[ln, col],   img[ln, col+1]],
#                 [img[ln+1, col-1], img[ln+1, col], img[ln+1, col+1]]
#             ], dtype=np.float32)
            
#             gx = np.sum(window * mx)
#             gy = np.sum(window * my)
            
#             new[ln, col] = np.sqrt(gx**2 + gy**2)
    
#     new = cv.normalize(new, None, 0, 255, cv.NORM_MINMAX)
#     return new.astype(np.uint8)


# def prewitt(img):
#     if len(img.shape) == 3:
#         img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
#     new = np.zeros_like(img, dtype=np.float32)
    

#     mx = np.array([[-1, 0, 1],
#                    [-2, 0, 2],
#                    [-1, 0, 1]], dtype=np.float32)

#     my = np.array([[-1, -2, -1],
#                    [ 0,  0,  0],
#                    [ 1,  2,  1]], dtype=np.float32)

#     h, w = img.shape

#     for ln in range(1, h-1):
#         for col in range(1, w-1):
#             window = np.array([
#                 [img[ln-1, col-1], img[ln-1, col], img[ln-1, col+1]],
#                 [img[ln, col-1],   img[ln, col],   img[ln, col+1]],
#                 [img[ln+1, col-1], img[ln+1, col], img[ln+1, col+1]]
#             ], dtype=np.float32)

#             gx = np.sum(window * mx)
#             gy = np.sum(window * my)

#             new[ln, col] = np.sqrt(gx**2 + gy**2)

#     new = cv.normalize(new, None, 0, 255, cv.NORM_MINMAX)
#     return new.astype(np.uint8)


# def img_bordas_branco(img, k=64):
#     img_quantizada = quantizacao_kmeans(img, k)

#     bordas_sobel = sobel(img_quantizada)

#     img_bordas_branco = np.ones_like(img) * 255  
#     mask_sobel = bordas_sobel > 50  
#     img_bordas_branco[mask_sobel] = [255, 0, 0]  

#     return img_bordas_branco


# def img_bordas_original(img, k=64):
#     img_quantizada = quantizacao_kmeans(img, k)

#     bordas_sobel = sobel(img_quantizada)

#     img_bordas_branco = np.ones_like(img) * 255
#     mask_sobel = bordas_sobel > 50
#     img_bordas_branco[mask_sobel] = [255, 0, 0] 

#     img_bordas_original = img_quantizada.copy()
#     img_bordas_original[mask_sobel] = [255, 0, 0] 

#     return img_bordas_original


# def bordas_sobel(img, k=64):
#     img_quantizada = quantizacao_kmeans(img, k)

#     bordas_sobel = sobel(img_quantizada)

#     return bordas_sobel

# def bordas_prewitt(img, k=64):
#     img_quantizada = quantizacao_kmeans(img, k)

#     bordas_prewitt = prewitt(img_quantizada)

#     return bordas_prewitt


# def img_quantizada(img, k=64):
#     img_quantizada = quantizacao_kmeans(img, k)

#     return img_quantizada


# def descritor_bic(img, threshold=50):
#     bordas = sobel(img)

#     img_original = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#     dados = img.reshape((-1, 3))
#     dados = np.float32(dados)
    
#     criterios = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     _, _, centros = cv.kmeans(dados, 64, None, criterios, 10, cv.KMEANS_RANDOM_CENTERS)
    
#     centros_quantizacao = np.uint8(centros)
    
#     mascara_bordas = bordas > threshold
    
#     n_cores = len(centros_quantizacao)
#     hist_bordas = np.zeros(n_cores)
#     hist_interior = np.zeros(n_cores)
    
#     altura, largura = img.shape[:2]
    
#     for i in range(altura):
#         for j in range(largura):
#             pixel = img[i, j]
#             distancias = np.sqrt(np.sum((centros_quantizacao - pixel)**2, axis=1))
#             idx_cor = np.argmin(distancias)
            
#             if mascara_bordas[i, j]:
#                 hist_bordas[idx_cor] += 1
#             else:
#                 hist_interior[idx_cor] += 1
    
#     hist_bordas = hist_bordas / np.sum(hist_bordas) if np.sum(hist_bordas) > 0 else hist_bordas
#     hist_interior = hist_interior / np.sum(hist_interior) if np.sum(hist_interior) > 0 else hist_interior
    
#     img_apenas_bordas = np.ones_like(img) * 255 
#     img_apenas_interior = np.ones_like(img) * 255  
    
#     for i in range(altura):
#         for j in range(largura):
#             if mascara_bordas[i, j]:
#                 img_apenas_bordas[i, j] = img[i, j]  
#             else:
#                 img_apenas_interior[i, j] = img[i, j]  
    
#     return hist_bordas
# # , hist_interior, img_apenas_bordas, img_apenas_interior, mascara_bordas