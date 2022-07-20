import sys
import cv2
import math
from PIL import ImageFilter
import numpy as np

BLUR_GRADE = 2.5
T = 150
GRID_radius = 1
SPLINE = True
curr_radius = 0
gradient_x, gradient_y = 0, 0


def paint(image, radius_l):
    canvas = np.zeros(image.shape)
    canvas.fill(255) # constant color canvas

    radius_l = sorted(radius_l, reverse=True)

    for radius in radius_l:
        print(radius)
        reference = gaussBlur(image, radius)
        canvas = paintLayer(canvas, reference, radius)
        print("done")
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
                brush_strokes.append(makeSplineStroke(x, y, reference, canvas, radius))
                if SPLINE == False:
                    x_error, y_error = np.unravel_index(region.argmax(), region.shape)

                    x_error = int(x-float(grid) /2) + x_error
                    y_error = int(y-float(grid) /2) + y_error

                    brush_strokes.append((radius, x_error, y_error, reference))

    if SPLINE == False:
        np.random.shuffle(brush_strokes)
        for r, x, y, ref in brush_strokes:
            makeStroke(canvas, radius, x, y, reference)
    else:
        np.random.shuffle(brush_strokes)
        for k, r, canvas, color in brush_strokes:
            draw_spline_stroke(k, r, canvas, color)

    return canvas

def makeSplineStroke(x0, y0, reference, canvas, radius):
    #minStrokes, maxStrokes = int(reference.shape[1] * 0.01), int(reference.shape[1] * 0.08)

    minStrokes, maxStrokes = 4, 16
    x, y = x0, y0
    ref_color = reference[y, x, :]
    k = [(x, y)]
    fc = 1
    lastDx, lastDy = 0, 0
    global curr_radius, gradient_x, gradient_y

    if curr_radius != radius:
    # https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
        depth = cv2.CV_64F
        gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

        if radius%2 == 0:
            kernel_radius = radius + 1
        else:
            kernel_radius = radius

        gradient_x = cv2.Sobel(gray_ref, depth, 1, 0, ksize=kernel_radius)
        gradient_y = cv2.Sobel(gray_ref, depth, 0, 1, ksize=kernel_radius)

        curr_radius = radius


    for i in range(1, maxStrokes):
        x = max(min(x, reference.shape[1]-1), 0)
        y = max(min(y, reference.shape[0]-1), 0)

        if i > minStrokes and (np.sum(np.abs(reference[y, x, :] - canvas[y, x, :])) < np.sum(np.abs(reference[y, x, :] - ref_color))):
            break

        if gradient_x[y, x]==0 and gradient_y[y, x]==0:
            break

        gx, gy = np.sum(gradient_x[y, x]), np.sum(gradient_y[y, x])
        dx, dy = -gy, gx

        if (lastDx * dx + lastDy * dy) < 0:
            dx, dy = -dx, -dy

        dx, dy = fc*dx + (1-fc)*lastDx, fc*dy + (1-fc)*lastDy

        if (pow(dx, 2) + pow(dy,2)) != 0:
           dx, dy = dx / pow(pow(dx, 2) + pow(dy, 2), 0.5), dy / pow(pow(dx, 2) + pow(dy, 2), 0.5)
        else:
            break

        x, y = int(x + radius*dx), int(y + radius*dy)
        lastDx, lastDy = dx, dy

        k.append((x,y))

    return (k, radius, canvas, ref_color)

def draw_spline_stroke(k, radius, canvas, color):
    b = int(color[0])
    g = int(color[1])
    r = int(color[2])

    color = (b/255, g/255, r/255)
    for point in k:
        x = point[0]
        y = point[1]

        cv2.circle(canvas, (x, y), int(radius), color, -1)


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
        