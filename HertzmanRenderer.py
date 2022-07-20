import sys
import cv2
import copy
import math, random
import time
from numba import jit, prange
from numba.experimental import jitclass
from PIL import ImageFilter
from color_processor import color_processor
import numpy as np
import PySimpleGUI as sg


class HertzmanRenderer:
    def __init__(self, path, brush_sizes, blur, threshold, curvature, grid_size, min_stroke, max_stroke, window, rgb_j, hsv_j, theme):
        self.path = path
        self.brush_sizes = brush_sizes
        self.blur = blur
        self.threshold = threshold
        self.grid_size = grid_size
        self.min_stroke = min_stroke
        self.max_stroke = max_stroke
        self.spline = True
        self.curr_radius = 0
        self.gradient_x = 0
        self.gradient_y = 0
        self.canvas_list = []
        self.window = window
        self.rgb_j = rgb_j
        self.hsv_j = hsv_j
        self.curvature = curvature
        self.totalTime = 0
        self.theme = theme
        if theme != "original":
            self.c_processor = color_processor(theme)

        self.progress = window["progress"]
        self.image = cv2.imread(path, cv2.IMREAD_COLOR)

        self.edge_image = self.generate_edges(self.image)

    def paint(self):
        start = time.time()
        canvas = np.zeros(self.image.shape)
        canvas.fill(255) # constant color canvas

        radius_l = sorted(self.brush_sizes, reverse=True)
        canvas_list = []
        prog_count = 0

        

        self.window["progress_log"].get()
        progress_size = 100 / len(radius_l) 

        first_frame = True

        for radius in radius_l:
            self.progressUpdate("\nRendering Layer with size: " + str(radius)) 
            reference = self.gaussBlur(self.image, radius)
            canvas = self.paintLayer(canvas, reference, radius, first_frame)
            canvas_list.append(copy.copy(canvas))
            prog_count += progress_size
            self.progress.update(current_count = prog_count)
            self.progressUpdate("\nDone Rendering Layer.")
            first_frame = False

        end = time.time()
        print("Time spent: ",end - start)

        return canvas_list

    def getOriginal(self):
        return self.image

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
        
    def paintLayer(self, canvas, reference, radius, first_frame):
        brush_strokes = []
        difference_image = self.elemDifference(canvas, reference)

        grid = int(self.grid_size * radius)

        start = time.time()
        for x in range(0, canvas.shape[1], grid):
            for y in range(0, canvas.shape[0], grid):
                region = difference_image[max(y-grid//2, 0):y+grid//2, max(x-grid//2, 0):x+grid//2]
                error = np.mean(region)

                if first_frame or (error > self.threshold):
                    brush_strokes.append(self.makeSplineStroke(x, y, reference, canvas, radius))
                    if self.spline == False:
                        x_error, y_error = np.unravel_index(region.argmax(), region.shape)

                        x_error = int(x-float(grid) /2) + x_error
                        y_error = int(y-float(grid) /2) + y_error

                        brush_strokes.append((radius, x_error, y_error, reference))
                        
        end = time.time()
        print("Time spent: ",end - start)
        print("MakeSpline: ", self.totalTime)

        if self.spline == False:
            np.random.shuffle(brush_strokes)
            for r, x, y, ref in brush_strokes:
                self.makeStroke(canvas, radius, x, y, reference)
        else:
            np.random.shuffle(brush_strokes)
            for k, r, canvas, color in brush_strokes:
                self.draw_spline_stroke(k, r, canvas, color)

        return canvas

    def makeSplineStroke(self, x0, y0, reference, canvas, radius):
        #minStrokes, maxStrokes = int(reference.shape[1] * 0.01), int(reference.shape[1] * 0.08)

        minStrokes, maxStrokes = self.min_stroke, self.max_stroke
        x, y = x0, y0


        if self.curr_radius != radius:
        # https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
            depth = cv2.CV_64F
            gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

            if radius%2 == 0:
                kernel_radius = radius + 1
            else:
                kernel_radius = radius

            self.gradient_x = cv2.Sobel(gray_ref, depth, 1, 0, ksize=kernel_radius)
            self.gradient_y = cv2.Sobel(gray_ref, depth, 0, 1, ksize=kernel_radius)

            self.curr_radius = radius

        start = time.time()

        k, self.gradient_x, self.gradient_y, ref_color = self.calcStroke(x, y, maxStrokes, minStrokes, reference, canvas, self.gradient_x, self.gradient_y, radius, self.curvature, self.edge_image)
        end = time.time() 
        self.totalTime += end - start

        return (k, radius, canvas, ref_color)

    @staticmethod
    @jit(nopython=True)
    def calcStroke(x, y, maxStrokes, minStrokes, reference, canvas, gradient_x, gradient_y, radius, fc, edge_image):
        ref_color = reference[y, x, :]
        k = [(x, y)]
        lastDx, lastDy = 0, 0
        prev_x, prev_y = x, y

        for i in range(1, maxStrokes):
            x = max(min(x, reference.shape[1]-1), 0)
            y = max(min(y, reference.shape[0]-1), 0)

            if i > minStrokes and (np.sum(np.abs(reference[y, x, :] - canvas[y, x, :])) < np.sum(np.abs(reference[y, x, :] - ref_color))):
                break

            if gradient_x[y, x]==0 and gradient_y[y, x]==0:
                break

            if edge_image[y, x] != 0:
                break

            dy, dx = gradient_x[y, x],-gradient_y[y, x]

            if (lastDx * dx + lastDy * dy) < 0:
                dx, dy = -dx, -dy

            dx, dy = fc*dx + (1-fc)*lastDx, fc*dy + (1-fc)*lastDy

            
            if (dx*dx + dy*dy) != 0:
                dx, dy = dx / pow(dx*dx + dy*dy, 0.5), dy / pow(dx*dx + dy*dy, 0.5)
   
            else:
                break
            
            x, y = int(x + radius*dx), int(y + radius*dy)
            prev_x, prev_y = x, y
            lastDx, lastDy = dx, dy

            k.append((x,y))

        return k, gradient_x, gradient_y, ref_color

    def draw_spline_stroke(self, k, radius, canvas, color):
        b = int(color[0])
        g = int(color[1])
        r = int(color[2])

        color = (b/255, g/255, r/255)


        if self.rgb_j[0] != 0 or self.rgb_j[1] != 0 or self.rgb_j[2] != 0:
            color = self.randomRGBJitter(color, (self.rgb_j[0], self.rgb_j[1], self.rgb_j[2]))

        if self.hsv_j[0] != 0 or self.hsv_j[1] != 0 or self.hsv_j[2] != 0:
            color = self.randomHSVJitter(color, (self.hsv_j[0], self.hsv_j[1], self.hsv_j[2]))

        if self.theme != 'original':
            color = self.c_processor.assign_color(color)

        for point in k:
            x = point[0]
            y = point[1]

            cv2.circle(canvas, (x, y), int(radius), color, -1)

    def randomRGBJitter(self, color, jitterValues):
        bgr_c = 0
        color_l = []

        for val in jitterValues:
            if val != 0:
                rJitter = random.randrange(-val*100, val*100, 1)
                new_c = color[bgr_c] + (rJitter / 100)

                if new_c < 0:
                    new_c = 0
                elif new_c > 1:
                    new_c = 1
            else:
                new_c = color[bgr_c]

            color_l.append(new_c)
            bgr_c = bgr_c + 1

        return (color_l[0], color_l[1], color_l[2])

    def randomHSVJitter(self, color, jitterValues):
        hsv_c = 0
        color_l = []

        color = np.uint8([[[color[0]*255, color[1]*255, color[2]*255]]])
        color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

        color = color[0][0]

        for val in jitterValues:
            if val != 0:
                rJitter = random.randrange(-val*100, val*100, 1)
                new_c = color[hsv_c] + (rJitter)

                if new_c < 0:
                    new_c = 0
                elif new_c > 255:
                    new_c = 255
            else:
                new_c = color[hsv_c]

            color_l.append(new_c)
            hsv_c = hsv_c + 1
        
        new_color = np.uint8([[[color_l[0], color_l[1], color_l[2]]]])
        bgr_color = cv2.cvtColor(new_color, cv2.COLOR_HSV2BGR)
        bgr_color = bgr_color[0][0]

        return (bgr_color[0] / 255, bgr_color[1] / 255, bgr_color[2] / 255)

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

    def generate_edges(self, image):
        edge_image = cv2.Canny(image, 250, 400)

        """cv2.imshow("Edge Image", edge_image)
        cv2.waitKey(0)"""

        return edge_image

"""if __name__ == '__main__':
    source_image = sys.argv[1]
    stroke_radius_l = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))

    renderer = HertzmanRenderer(source_image, stroke_radius_l, 2.5, 100, 1, 4, 16)
    renderer.paint()"""