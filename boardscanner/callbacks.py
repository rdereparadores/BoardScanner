import cv2
import numpy as np

from boardscanner.Config import Config
from boardscanner.Utils import Utils

manual_board_last_point_selected = 2

def mouse_callback(event, x, y, flags, param):
    global manual_board_last_point_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        # set_hsv_color_callback
        frame = Utils.capture_image_from_camera()
        bgr = frame[y, x]
        bgr = np.uint8([[ bgr ]])
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
        Utils.set_hsv_color(hsv)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # set_manual_board_points_callback
        if manual_board_last_point_selected == 2:
            Config.MANUAL_BOARD_POINT1 = (x, y)
            manual_board_last_point_selected = 1
        else:
            Config.MANUAL_BOARD_POINT2 = (x, y)
            manual_board_last_point_selected = 2

def set_hsv_tolerance_callback(value):
    Utils.set_hsv_tolerance(value)

def set_word_dilate_ksize_callback(value):
    Config.WORD_DILATE_KSIZE = value