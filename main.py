import os
import sys
import cv2 as cv
import image_processing as imp

fmts = ".jpg", ".png", ".jpeg"
argv = sys.argv[1:] or [a for a in os.listdir() if a.endswith(fmts)]

if not argv:
    print("⚠️  Usage: python main.py [path/to/image1.jpg image2.png ...]")
    print("ℹ️  Or bring images to the current directory")
    print("ℹ️  Supported formats:", ", ".join(fmts))
    sys.exit(0)

funcs = *filter(callable, imp.__dict__.values()),

for src in argv:
    # try:
    img = cv.imread(src)
    # except:
    #     # print("⚠️  Error reading", src + ":", e)
    #     img = cv.imread(src, cv.IMREAD_GRAYSCALE)

    if img is None:
        continue

    src = src.rsplit(".", 1)

    for f in funcs:
        dst = f"-{f.__name__}.".join(src)
        new = f(img)
        if new is None:
            continue
        if cv.imwrite(dst, new):
            print("✅ Saved", dst)
