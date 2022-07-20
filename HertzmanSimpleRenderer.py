import sys
import cv2
import copy
import math
from PIL import ImageFilter
import numpy as np

class HertzmanSimpleRenderer:
    def __init__(self, path, brush_sizes, blur, threshold, grid_size, window):
        self.path = path
        self.brush_sizes = brush_sizes
        self.blur = blur
        self.threshold = threshold
        self.grid_size = grid_size
        self.canvas_list = []
        self.window = window

        self.progress = window["progress"]

        self.image = cv2.imread(path, cv2.IMREAD_COLOR)

    def paint(self):
        canvas = np.zeros(self.image.shape)
        canvas.fill(255) # constant color canvas

        canvas_list = []
        prog_count = 0

        self.window["progress_log"].get()
        
        radius_l = sorted(self.brush_sizes, reverse=True)
        progress_size = 100 / len(radius_l) 

        for radius in radius_l:
            self.progressUpdate("\nRendering Layer with size: " + str(radius)) 
            reference = self.gaussBlur(self.image, radius)
            canvas = self.paintLayer(canvas, reference, radius)
            canvas_list.append(copy.copy(canvas))
            prog_count += progress_size
            self.progress.update(current_count = prog_count)
            self.progressUpdate("\nDone Rendering Layer.")

        return canvas_list

    def progressUpdate(self, text):
        current = self.window["progress_log"].get()
        line_count = current.count("\n")

        if line_count > 8:
            current = current[current.find("\n")+2:] + text
        else:
            current = current + text

        self.window["progress_log"].update(current)

    def gaussBlur(self, image, radius):
        radius = int(radius*self.blur)
        if (radius%2) != 1:
            radius += 1

        reference = cv2.GaussianBlur(image, (radius, radius), 0)
        return reference

    def getOriginal(self):
        return self.image

    def paintLayer(self, canvas, reference, radius):
        brush_strokes = []
        difference_image = self.elemDifference(canvas, reference)

        grid = int(self.grid_size * radius)

        for x in range(0, canvas.shape[1], grid):
            for y in range(0, canvas.shape[0], grid):
                region = difference_image[max(y-grid//2, 0):y+grid//2, max(x-grid//2, 0):x+grid//2]
                error = np.mean(region)

                if error > self.threshold:
                    x_error, y_error = np.unravel_index(region.argmax(), region.shape)

                    x_error = int(x-float(grid) /2) + x_error
                    y_error = int(y-float(grid) /2) + y_error

                    brush_strokes.append((radius, x_error, y_error, reference))

        np.random.shuffle(brush_strokes)
        for r, x, y, ref in brush_strokes:
            self.makeStroke(canvas, radius, x, y, reference)

        return canvas


    def makeStroke(self, canvas, radius, err_x, err_y, reference):
        if err_x < reference.shape[1] and err_y < reference.shape[0]: 
            b = int(reference[err_y, err_x, 0])
            g = int(reference[err_y, err_x, 1])
            r = int(reference[err_y, err_x, 2])

            color = (b/255, g/255, r/255)
            ny = err_y + int(-math.sin((225 * math.pi) / 180) * 3)
            nx = err_x + int(math.cos((225 * math.pi) / 180) * 3)
            cv2.line(canvas, (err_x, err_y), (nx, ny), color, int(radius))


    def elemDifference(self, canvas, reference):
        difference = np.sum(np.abs(canvas - reference), axis=2)

        return difference


"""if __name__ == '__main__':
    source_image = sys.argv[1]
    stroke_radius_l = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))

    image = cv2.imread(source_image, cv2.IMREAD_COLOR)
    paint(image, stroke_radius_l)"""
        