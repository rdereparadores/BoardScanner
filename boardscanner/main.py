import cv2
from BoardProcessingUtils import BoardProcessingUtils
from Config import Config

# Fase 1: Obtener la región de la pizarra
while True:
    frame = BoardProcessingUtils.capture_image_from_camera()
    hsv = BoardProcessingUtils.image_to_filtered_hsv(frame)
    canny = BoardProcessingUtils.canny_with_mask(frame, hsv['mask'])
    contour = BoardProcessingUtils.get_contour_from_canny(canny)

    cv2.imshow('Frame', frame)
    cv2.imshow('HSV', hsv['hsv'])
    cv2.imshow('Canny', canny)

    key_pressed = cv2.waitKey(100)
    if key_pressed & 0xFF == ord('q'):
        break


Config.CAMERA.release()
cv2.destroyAllWindows()