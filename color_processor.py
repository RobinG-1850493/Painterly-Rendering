import cv2 as cv
import numpy as np

class color_processor:
    def __init__(self, theme):
        self.palette = self.load_palette(theme)

    def load_palette(self, theme):
        pal_img = cv.imread(theme)
        hsv_img = cv.cvtColor(pal_img, cv.COLOR_BGR2HSV)

        return hsv_img

    def assign_color(self, color):

        # convert color to HSV
        color = np.uint8([[[color[0]*255, color[1]*255, color[2]*255]]])
        color = cv.cvtColor(color, cv.COLOR_BGR2HSV)
        color = color[0][0]

        new_hue = self.palette[0, color[0]]
        print(new_hue[0])

        new_color = np.uint8([[[new_hue[0], color[1], color[2]]]])
        bgr_color = cv.cvtColor(new_color, cv.COLOR_HSV2BGR)
        bgr_color = bgr_color[0][0]



        return (bgr_color[0] / 255, bgr_color[1] / 255, bgr_color[2] / 255)

