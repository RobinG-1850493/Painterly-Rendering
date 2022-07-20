import numpy as np
import cv2 as cv


img = cv.imread('C:\\Users\\Robin\\Desktop\\inf-masterproef-21-22-student-RobinG-1850493\\code\\Painterly Rendering\\images\\landscape.jpg',0)
edges = cv.Canny(img, 200,250)

cv.imshow("title", edges)
cv.waitKey(0)

