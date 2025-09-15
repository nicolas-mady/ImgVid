import numpy as np

def bright(img, factor=1.2):
    return np.uint8(np.clip(img * factor, 0, 255))

def grey(img):
    return np.uint8(img @ [.1, .6, .3])

def negative(img):
    return 255 - img
