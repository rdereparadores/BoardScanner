import datetime

import cv2
import keras.models
import numpy as np
from boardscanner.Config import Config
from boardscanner.Globals import Globals
import mediapipe as mp

class Utils:

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
    def morphological_closing(canny):
        # Aplica el filtro morfológico de cierre (erosión + dilatación)
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def get_contour(canny):
        # Detecta los contornos cerrados de la imagen
        contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Ordena todos los contornos encontrados según su área, de mayor a menor
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Escoge, de todos los contornos, el de mayor área con 4 lados
        board_contour = None
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
            if len(approx) == 4:
                board_contour = approx
                break
        return board_contour

    @staticmethod
    def order_rectangle_points(points):
        """
        Ordena los puntos de un rectángulo en el orden: top-left, top-right, bottom-right, bottom-left

        :param points: Array de 4 puntos que definen el rectángulo
        :return: Array con los 4 puntos ordenados
        """
        points = points.reshape(4, 2)
        rect = np.zeros((4, 2), dtype='float32')

        # Suma para cada punto los valores X e Y
        # El punto (0, 0) en OpenCV se encuentra en top-left
        # top-left -> menor de los valores
        # bottom-right -> mayor de los valores
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]

        # Resta para cada punto los valores X e Y
        # top-right -> menor de los valores
        # bottom-left -> mayor de los valores
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]

        return rect

    @staticmethod
    def get_transformed_rectangle_region(image, points):
        """
        Extrae de la imagen origen la región delimitada por el rectángulo y la devuelve con su perspectiva transformada.

        :param image: Imagen origen
        :param points: Array de 4 puntos que definen el rectángulo
        :return: Región de la imagen origen con la perspectiva transformada
        """
        rect = Utils.order_rectangle_points(points)
        tl, tr, br, bl = rect

        # Calcula los dos anchos del rectángulo, y escoge el mayor
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Calcula los dos altos del rectángulo, y escoge el mayor
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        new_rect = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # Matriz de transformación de perspectiva
        M = cv2.getPerspectiveTransform(rect, new_rect)

        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    @staticmethod
    def get_word_regions(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,
            C=10
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (Config.WORD_DILATE_KSIZE, Config.WORD_DILATE_KSIZE))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        word_contours = []
        word_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area > Config.WORD_MIN_AREA and w > h * 0.5:
                word_contours.append((x, y, w, h))

                x, y, w, h = cv2.boundingRect(contour)

                word_region = image[y:y + h, x:x + w]
                word_regions.append(word_region)

        return {
            "binary": binary,
            "dilated": dilated,
            "contours": word_contours,
            "regions": word_regions
        }

    @staticmethod
    def preprocess_character_emnist(character):
        gray = cv2.cvtColor(character, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        resized = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)
        resized = resized.astype(np.float32) / 255.0
        resized = resized.reshape(1, 28, 28, 1)
        return resized


    @staticmethod
    def get_character_emnist(character):
        pred = Globals.emnist_model.predict(character, verbose=0)
        clase = np.argmax(pred)

        return Globals.emnist_mappings.get(clase)

    @staticmethod
    def draw_string_to_image(image, contour, string: str):
        result = image.copy()
        x, y, w, h = contour
        cv2.rectangle(result, (x, y), (x+w, y+h), (128, 128, 128), -1)

        (text_width, text_height), baseline = cv2.getTextSize(string, cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)

        text_x = x + (w - text_width) // 2
        text_y = y + (h + text_height) // 2

        cv2.putText(result, string, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)
        return result


    @staticmethod
    def load_emnist_model():
        model = keras.models.load_model('boardscanner/emnist_sample.keras')
        Globals.emnist_model = model

        # Cargar mappings
        mapping = {}
        with open('./emnist/emnist-byclass-mapping.txt', 'r') as f:
            for line in f:
                parts = line.strip().split()
                num, char = parts
                mapping[int(num)] = chr(int(char))
        Globals.emnist_mappings = mapping

    @staticmethod
    def load_emnist_knn_model():
        data = np.loadtxt('./emnist/emnist-balanced-train.csv', delimiter=',', dtype=np.uint8)
        y_train = data[:, 0].astype(np.float32)
        x_train = data[:, 1:]

        def fix_orientation(img_flat):
            img = img_flat.reshape(28, 28)
            img = np.flip(img, axis=0)
            img = np.rot90(img, k=-1)
            return img

        x_train = np.array([fix_orientation(x) for x in x_train])

        x_train_flat = x_train.reshape(len(x_train), -1).astype(np.float32) / 255.0

        knn = cv2.ml.KNearest.create()
        knn.train(x_train_flat, cv2.ml.ROW_SAMPLE, y_train)

        Globals.emnist_knn_model = knn

    @staticmethod
    def set_hsv_color(hsv):
        h, s, v = hsv
        lower_h = max(0, int(h) - Config.HSV_TOLERANCE)
        upper_h = min(255, int(h) + Config.HSV_TOLERANCE)
        Config.HSV_LOWER = np.array([lower_h, 50, 0])
        Config.HSV_UPPER = np.array([upper_h, 255, 255])
        Config.HSV_COLOR = np.array([h, s, v])

    @staticmethod
    def set_hsv_tolerance(value):
        Config.HSV_TOLERANCE = value
        Utils.set_hsv_color(Config.HSV_COLOR)

    @staticmethod
    def get_bounding_rect(point1, point2):
        x1, y1 = point1
        x2, y2 = point2

        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)

        return np.array([
            [[x_min, y_min]],
            [[x_max, y_min]],
            [[x_max, y_max]],
            [[x_min, y_max]]
        ])

    @staticmethod
    def get_manual_board_contour():
        return Utils.get_bounding_rect(Config.MANUAL_BOARD_POINT1, Config.MANUAL_BOARD_POINT2)

    @staticmethod
    def get_words_google(img):
        from google.cloud import vision

        _, img_jpg = cv2.imencode('.jpg', img)
        byte_img = img_jpg.tobytes()
        google_img = vision.Image(content=byte_img)

        client = vision.ImageAnnotatorClient()
        response = client.text_detection(image=google_img)

        words = []
        for word in response.text_annotations[1:]:
            vertices = word.bounding_poly.vertices

            x_coords = [v.x for v in vertices]
            y_coords = [v.y for v in vertices]
            x = min(x_coords)
            y = min(y_coords)
            w = max(x_coords) - x
            h = max(y_coords) - y
            data = {
                'word': word.description,
                'contour': (x, y, w, h)
            }
            words.append(data)
        return words

    @staticmethod
    def is_palm_open(hand_landmarks):
        landmarks = hand_landmarks.landmark
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        extended = []
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y < landmarks[pip].y:
                extended.append(True)
            else:
                extended.append(False)

        z_values = [landmarks[i].z for i in finger_tips]
        z_diff = np.std(z_values)

        return all(extended) and z_diff < 0.02

    @staticmethod
    def detect_gesture(img):
        output = img.copy()

        with Globals.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        ) as hands:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            gesture_detected = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    Globals.mp_draw.draw_landmarks(output, hand_landmarks, Globals.mp_hands.HAND_CONNECTIONS)
                    if Utils.is_palm_open(hand_landmarks):
                        gesture_detected = True
        return {
            'frame': output,
            'result': gesture_detected
        }