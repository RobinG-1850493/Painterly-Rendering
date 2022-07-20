import sys
import cv2
import copy
import math, random
import time
from numba import jit, cuda, float32, vectorize
from numba.experimental import jitclass
from PIL import ImageFilter
import numpy as np
import PySimpleGUI as sg


class HertzmanVideoRenderer:
    def __init__(self, path, brush_sizes, blur, threshold, curvature, grid_size, min_stroke, max_stroke, window, rgb_j, hsv_j):
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

        self.progress = window["progress"]

    def paint(self):
        start = time.time()

        video_frames = []
        radius_l = sorted(self.brush_sizes, reverse=True)
        prog_count = 0

        first_frame = True

        print(self.path)
        cap = cv2.VideoCapture(self.path)

        #frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.window["progress_log"].get()
        progress_size = 100 / 4

        while cap.isOpened():
            ret, frame = cap.read()
            
            if first_frame:
                canvas = np.zeros(frame.shape)
                canvas.fill(255) # constant color canvas

            if ret == True:
                for radius in radius_l:
                    self.progressUpdate("\nRendering Layer with size: " + str(radius)) 
                    reference = self.gaussBlur(frame, radius)
                    canvas = self.paintLayer(canvas, reference, radius, first_frame)
                    video_frames.append(copy.copy(canvas))
                    prog_count += progress_size
                    self.progress.update(current_count = prog_count)
                    self.progressUpdate("\nDone Rendering Layer.")
                    first_frame = False

                self.displayImage(canvas, self.window)

        cap.release()
        end = time.time()
        print("Time spent: ",end - start)

        return canvas

    def displayImage(self, canvas, window):
        graph = window["-main_canvas-"]

        canvas = canvas * 255
        canvas = canvas.astype("uint8")
        #canvas = canvas[:,:,::-1]

        canvas = cv2.resize(canvas, (graph.get_size()[0], graph.get_size()[1]), interpolation = cv2.INTER_NEAREST)

        _, im_arr = cv2.imencode('.png', canvas)
        img_str = im_arr.tobytes()

        graph.draw_image(data=img_str, location=(0, 600))

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

        k, self.gradient_x, self.gradient_y, ref_color = self.calcStroke(x, y, maxStrokes, minStrokes, reference, canvas, self.gradient_x, self.gradient_y, radius, self.curvature)
        end = time.time() 
        self.totalTime += end - start

        return (k, radius, canvas, ref_color)

    @staticmethod
    @jit(nopython=True)
    def calcStroke(x, y, maxStrokes, minStrokes, reference, canvas, gradient_x, gradient_y, radius, fc):
        ref_color = reference[y, x, :]
        k = [(x, y)]
        lastDx, lastDy = 0, 0

        for i in range(1, maxStrokes):
            x = max(min(x, reference.shape[1]-1), 0)
            y = max(min(y, reference.shape[0]-1), 0)

            if i > minStrokes and (np.sum(np.abs(reference[y, x, :] - canvas[y, x, :])) < np.sum(np.abs(reference[y, x, :] - ref_color))):
                break

            if gradient_x[y, x]==0 and gradient_y[y, x]==0:
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


    def elemDifference(self, canvas, reference):
        difference = np.sum(np.abs(canvas - reference), axis=2)

        return difference