import os
import sys

import cv2

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
    img = cv2.imread(src)

    if img is None:
        continue

    src = src.rsplit('.', 1)

    for f in funcs:
        dst = f"_({f.__name__}).".join(src)
        if cv2.imwrite(dst, f(img)):
            print("✅ Saved", dst)
