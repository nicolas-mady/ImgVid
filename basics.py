import numpy as np
import cv2 as cv


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


def global_hist(img, nome="q3.txt", *, ret=False):
    hist = np.zeros(768, dtype=np.uint32)
    for row in img:
        for col in row:
            b, g, r = map(int, col)
            hist[r] += 1
            hist[g + 256] += 1
            hist[b + 512] += 1
    if ret:
        return hist
    with open(nome, "w") as f:
        f.write(f'{hist}\n')
    print(f"✅ Saved {nome}")


def local_hist(img):
    h = img.shape[0] // 3
    pts = [img[i*h:(i+1)*h] for i in range(3)]
    cv.imwrite("top.jpg", pts[0])
    cv.imwrite("mid.jpg", pts[1])
    cv.imwrite("bot.jpg", pts[2])
    histograms = []
    for pt in pts:
        histograms += global_hist(pt, ret=True).tolist()
    hist = np.array(histograms)
    with open("q4.txt", "w") as f:
        np.savetxt(f, hist, fmt='%d', newline=' ')
    print(f"✅ Saved q4.txt")