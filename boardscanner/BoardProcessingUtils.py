import cv2
from Config import Config

class BoardProcessingUtils:

    @staticmethod
    def capture_image_from_camera():
        # Captura una imagen desde la cámara
        success, frame = Config.CAMERA.read()

        if not success:
            print('Error al capturar desde la cámara')
            return None

        return frame

    @staticmethod
    def image_to_filtered_hsv(img):
        # Convierte la imagen de entrada de BGR a HSV
        hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Crea una máscara para filtrar de la imagen únicamente los colores que se encuentren en el rango
        mask = cv2.inRange(hsv_frame, Config.HSV_LOWER, Config.HSV_UPPER)
        return {
            'hsv': hsv_frame,
            'mask': mask
        }

    @staticmethod
    def canny_with_mask(img, mask):
        # Aplica la máscara a la imagen
        masked = cv2.bitwise_and(img, img, mask=mask)
        # Transforma la imagen de BGR a escala de grises
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        # Aplica un filtro gaussiano
        blur = cv2.GaussianBlur(gray, (Config.GAUSSIAN_BLUR_KSIZE, Config.GAUSSIAN_BLUR_KSIZE), 0)
        # Ejecuta el algoritmo de detección de bordes Canny
        edges = cv2.Canny(blur, Config.CANNY_THRESHOLD1, Config.CANNY_THRESHOLD2)
        return edges

    @staticmethod
    def get_contour_from_canny(canny):
        # Detecta los contornos cerrados de la imagen
        contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Ordena todos los contornos encontrados según su área, de mayor a menor
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Escoge, de todos los contornos, el de mayor área de 4 lados
        board_contour = None
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
            if len(approx) == 4:
                board_contour = approx
                break

        return board_contour