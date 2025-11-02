import cv2
import numpy as np

class Config:
    WINDOW_NAME = 'Pizarra'
    DEVICE_ID = 0
    CAMERA = cv2.VideoCapture(DEVICE_ID)

    MANUAL_BOARD_POINT1 = (0, 0)
    MANUAL_BOARD_POINT2 = (50, 50)

    HSV_TOLERANCE = 30
    HSV_COLOR = np.array([0, 0, 0])
    HSV_LOWER = np.array([0, 50, 0])
    HSV_UPPER = np.array([0, 255, 255])

    CANNY_THRESHOLD1 = 50
    CANNY_THRESHOLD2 = 150

    GAUSSIAN_BLUR_KSIZE = 5

    WORD_MIN_AREA = 200
    WORD_DILATE_KSIZE = 5

    HAND_GESTURE_TRIGGER = 60