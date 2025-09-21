from inspect import isfunction

import cv2 as cv

import basics
import radiometrics
import filters
import borders
import bic

for src, mod in (
    ('im0.jpg', basics),
    ('im1.jpg', radiometrics),
    ('im2.jpg', filters),
    ('im3.jpg', borders),
    ('im3.jpg', bic),
):
    img = cv.imread(src)

    if img is None:
        print('❌ Could not read', src)
        continue

    src = src.rsplit('.', 1)

    for f in filter(isfunction, mod.__dict__.values()):
        res = f(img)
        if not isinstance(res, tuple):
            res = (res,)
        sf = ''
        for new in res:
            if new is None:
                continue
            dst = f'-{f.__name__}{sf}.'.join(src)
            # dst = 't0.jpg'
            if cv.imwrite(dst, new):
                print('✅ Saved', dst)
            sf = '-bordas'