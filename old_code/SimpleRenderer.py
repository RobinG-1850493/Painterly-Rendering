import sys
import math
import cv2
import numpy as np


class renderer:
    def __init__(self, source, stroke_list):
        self.source = source
        self.stroke_list = stroke_list
        self.blurGrade = 5
        self.threshold = 100
        
    def paint(self):
        image_canvas = np.zeros(self.source.shape)
        image_canvas.fill(255)

        for r in self.stroke_list:
            ref = self.blur(r)
            image_canvas = self.render_layer(image_canvas, ref, r)
            

    def render_layer(self, image_canvas, ref, r):
        strokes = []
        grid_size = 1 * r

        diff = self.difference(image_canvas, ref)

        for x in range(0, image_canvas.shape[1], grid_size):
            for y in range(0, image_canvas.shape[0], grid_size):
                region = diff[x-grid_size//2:x+grid_size//2, y-grid_size//2:y+grid_size//2]

                error = np.sum(region) / np.power(grid_size, 2)
                

                if (error > self.threshold):
                    #print(error)
                    x_error, y_error, temp = np.unravel_index(region.argmax(), region.shape)

                    x_error = int(x-float(grid_size) /2) + x_error
                    y_error = int(y-float(grid_size) /2) + y_error

                    strokes.append((r, x_error, y_error, ref))
        
        np.random.shuffle(strokes)
        for r, x, y, ref in strokes:
            self.make_stroke(image_canvas, r, x, y, ref)

        self.showImage(image_canvas)
        return image_canvas

    def make_stroke(self, image_canvas, radius, x, y, ref):
        b = int(ref[y, x, 0])
        g = int(ref[y, x, 1])
        r = int(ref[y, x, 2])

        color = (b/255, g/255, r/255)
        ny = y + int(-math.sin((225 * math.pi) / 180) * 3)
        nx = x + int(math.cos((225 * math.pi) / 180) * 3)
        cv2.line(image_canvas, (x, y), (nx, ny), color, radius)

    def blur(self, radius):
        ref = cv2.GaussianBlur(self.source, (radius * self.blurGrade,radius * self.blurGrade), 0)

        return ref

    def difference(self, image_canvas, ref):
        """
        canv_r, canv_g, canv_b = np.split(image_canvas, 3, 2)
        ref_r, ref_g, ref_b = np.split(ref, 3, 2)

        n_diff = np.clip(np.abs(np.power(canv_r - ref_r, 2) + np.power(canv_g - ref_g, 2) + np.power(canv_b - ref_b, 2)), 0, 255)
        n_diff = np.squeeze(n_diff)
        """

        diff = np.clip(abs(image_canvas - ref),0,255)

        return diff

    def showImage(self, im):
        cv2.imshow("Image", im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    source = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    strokes = [3,5,7]

    painter = renderer(source, strokes)
    painter.paint()