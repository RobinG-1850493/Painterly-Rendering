import sys
import cv2
import math
from PIL import ImageFilter
import numpy as np

BLUR_GRADE = 2.5
T = 100
GRID_radius = 0.5



def paint(image, radius_l):
    canvas = np.zeros(image.shape)
    canvas.fill(255) # constant color canvas

    radius_l = sorted(radius_l, reverse=True)

    for radius in radius_l:
        print(radius)
        reference = gaussBlur(image, radius)
        paintLayer(canvas, reference, radius)

        cv2.imshow("test", canvas)
        cv2.waitKey(0)


def gaussBlur(image, radius):
    radius = int(radius*BLUR_GRADE)
    if (radius%2) != 1:
        radius += 1

    reference = cv2.GaussianBlur(image, (radius, radius), 0)
    return reference

def paintLayer(canvas, reference, radius):
    brush_strokes = []
    difference_image = elemDifference(canvas, reference)

    grid = int(GRID_radius * radius)

    for x in range(0, canvas.shape[1], grid):
        for y in range(0, canvas.shape[0], grid):
            region = difference_image[max(y-grid//2, 0):y+grid//2, max(x-grid//2, 0):x+grid//2]
            error = np.mean(region)

            if error > T:
                x_error, y_error = np.unravel_index(region.argmax(), region.shape)

                x_error = int(x-float(grid) /2) + x_error
                y_error = int(y-float(grid) /2) + y_error

                brush_strokes.append((radius, x_error, y_error, reference))

    np.random.shuffle(brush_strokes)
    for r, x, y, ref in brush_strokes:
        makeStroke(canvas, radius, x, y, reference)

    return canvas


def makeStroke(canvas, radius, err_x, err_y, reference):
    if err_x < reference.shape[1] and err_y < reference.shape[0]: 
        b = int(reference[err_y, err_x, 0])
        g = int(reference[err_y, err_x, 1])
        r = int(reference[err_y, err_x, 2])

        color = (b/255, g/255, r/255)
        ny = err_y + int(-math.sin((225 * math.pi) / 180) * 3)
        nx = err_x + int(math.cos((225 * math.pi) / 180) * 3)
        cv2.line(canvas, (err_x, err_y), (nx, ny), color, int(radius))


def elemDifference(canvas, reference):
    difference = np.sum(np.abs(canvas - reference), axis=2)

    return difference


if __name__ == '__main__':
    source_image = sys.argv[1]
    stroke_radius_l = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))

    image = cv2.imread(source_image, cv2.IMREAD_COLOR)
    paint(image, stroke_radius_l)
        