import cv2
import numpy as np

class Config:
    WINDOW_NAME = 'Pizarra'
    DEVICE_ID = 0
    CAMERA = cv2.VideoCapture(DEVICE_ID)

    HSV_TOLERANCE = 0
    #HSV_TOLERANCE_H_RATIO = 0.15
    #HSV_TOLERANCE_S_RATIO = 0.4
    #HSV_TOLERANCE_V_RATIO = 0.45
    HSV_COLOR = np.array([0, 0, 0])
    HSV_LOWER = np.array([0, 150, 0])
    HSV_UPPER = np.array([0, 255, 255])

    CANNY_THRESHOLD1 = 50
    CANNY_THRESHOLD2 = 150

    GAUSSIAN_BLUR_KSIZE = 5
    OUTPUT_FONT_SIZE_MULTIPLIER = 0.55
    OUTPUT_BACKGROUND_BOX_MULTIPLIER = 1.02