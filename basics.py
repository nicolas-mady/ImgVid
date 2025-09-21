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


def local_hist(img, k=3, o='h', nome='q4.txt'):
    if o == 'h':
        h = img.shape[0] // k
        pts = [img[i*h:(i+1)*h] for i in range(k)]
    else:
        w = img.shape[1] // k
        pts = [img[:, i*w:(i+1)*w] for i in range(k)]
    hist = []
    for pt in pts:
        hist += global_hist(pt, ret=True)
    with open(nome, "w") as f:
        f.write(str(hist))
    print(f"✅ Saved {nome}")