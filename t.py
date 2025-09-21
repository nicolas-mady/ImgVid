import cv2 as cv
import sys
import borders
import basics
import radiometrics
import bic

im_interior, im_bordas = bic.bic(cv.imread(sys.argv[1]))
cv.imwrite("t0.jpg", im_interior)