import datetime

from boardscanner.Globals import Globals
from boardscanner.callbacks import *

def main_loop():
    ocr_mode = 'none'
    use_google_vision = False
    manual_board_contour = False
    hand_gesture_count = 0

    cv2.imshow(Config.WINDOW_NAME, Utils.capture_image_from_camera())
    cv2.setMouseCallback(Config.WINDOW_NAME, mouse_callback)
    cv2.createTrackbar('Tolerancia del filtro HSV: ', Config.WINDOW_NAME, 30, 200, set_hsv_tolerance_callback)
    cv2.createTrackbar('Ksize para la dilatación de palabras: ', Config.WINDOW_NAME, 5, 30, set_word_dilate_ksize_callback)

    # Cargar los modelos para OCR
    Utils.load_emnist_model()
    Utils.load_emnist_knn_model()

    while True:
        # Fase 1: Obtener la región de la pizarra
        frame = Utils.capture_image_from_camera()
        hsv = Utils.image_to_filtered_hsv(frame)
        canny = Utils.canny_with_mask(frame, hsv['mask'])
        closure = Utils.morphological_closing(canny)
        if manual_board_contour:
            contour = Utils.get_manual_board_contour()
        else:
            contour = Utils.get_contour(closure)
        board = None
        board_with_word_contours = None
        board_adaptive_threshold = None
        board_dilated = None

        emnist_mlp_frame = frame.copy()
        emnist_knn_frame = frame.copy()
        google_vision_frame = Globals.google_vision_frame
        if google_vision_frame is None:
            google_vision_frame = frame.copy()

        if contour is not None:
            board = Utils.get_transformed_rectangle_region(frame, contour)

            word_extract = Utils.get_word_regions(board)
            board_adaptive_threshold = word_extract['binary']
            board_dilated = word_extract['dilated']
            board_with_word_contours = board.copy()

            for (x, y, w, h) in word_extract['contours']:
                cv2.rectangle(board_with_word_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Fase 2.2: Reconocimiento con kNN
            if ocr_mode == 'knn':
                emnist_knn_frame = board.copy()
                for i, region in enumerate(word_extract['regions']):
                    char = Utils.preprocess_character_emnist(region)
                    char_2d = np.array(char, dtype=np.float32).reshape(1, -1)
                    ret, result, neighbours, dist = Globals.emnist_knn_model.findNearest(char_2d, k=3)
                    emnist_knn_frame = Utils.draw_string_to_image(emnist_knn_frame, word_extract['contours'][i], Globals.emnist_mappings.get(int(result[0])))

            # Fase 2.2: Reconocimiento con MLP
            if ocr_mode == 'mlp':
                emnist_mlp_frame = board.copy()
                for i, region in enumerate(word_extract['regions']):
                    char = Utils.preprocess_character_emnist(region)
                    result = Utils.get_character_emnist(char)
                    emnist_mlp_frame = Utils.draw_string_to_image(emnist_mlp_frame, word_extract['contours'][i], str(result))

            # Fase 2.3: Reconocimiento con Google Vision API
            if use_google_vision:
                google_vision_frame = board.copy()
                words = Utils.get_words_google(board)
                for word in words:
                    google_vision_frame = Utils.draw_string_to_image(google_vision_frame, word['contour'], word['word'])
                use_google_vision = False

        # Fase 3: Reconocimiento de gestos
        gesture = Utils.detect_gesture(frame)
        cv2.imshow('prueba', gesture['frame'])
        if gesture['result']:
            hand_gesture_count += 1
            print(hand_gesture_count)
            if hand_gesture_count == Config.HAND_GESTURE_TRIGGER:
                if ocr_mode == 'knn':
                    cv2.imwrite('save' + str(datetime.datetime.now()) + '.png', emnist_knn_frame)
                    cv2.putText(emnist_knn_frame, "Imagen guardada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Imagen guardada', emnist_knn_frame)
                elif ocr_mode == 'mlp':
                    cv2.imwrite('save' + str(datetime.datetime.now()) + '.png', emnist_mlp_frame)
                    cv2.putText(emnist_mlp_frame, "Imagen guardada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Imagen guardada', emnist_mlp_frame)
                elif ocr_mode == 'google':
                    cv2.putText(google_vision_frame, "Imagen guardada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Imagen guardada', google_vision_frame)
                else:
                    cv2.putText(frame, "Imagen guardada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Imagen guardada', frame)
                hand_gesture_count = 0
        else:
            hand_gesture_count = 0


        # Mostrar los resultados
        frame_with_contour = None
        if contour is not None:
            frame_with_contour = frame.copy()
            cv2.drawContours(frame_with_contour, [contour], 0, (0, 255, 0), 4)
            cv2.imshow(Config.WINDOW_NAME, frame_with_contour)
        else:
            cv2.imshow(Config.WINDOW_NAME, frame)

        with Globals.lock:
            Globals.original_frame = frame.copy()
            Globals.hsv_frame = hsv['hsv'].copy()
            Globals.hsv_mask_frame = hsv['mask'].copy()
            Globals.canny_frame = canny.copy()
            Globals.closure_frame = closure.copy()
            if contour is not None:
                Globals.contour_frame = frame_with_contour.copy()
                Globals.board_frame = board.copy()

                Globals.board_adaptive_threshold_frame = board_adaptive_threshold.copy()
                Globals.board_dilated_frame = board_dilated.copy()
                Globals.board_with_word_contours_frame = board_with_word_contours.copy()

                Globals.emnist_knn_frame = emnist_knn_frame.copy()
                Globals.emnist_mlp_frame = emnist_mlp_frame.copy()
                Globals.google_vision_frame = google_vision_frame.copy()
            else:
                Globals.contour_frame = frame.copy()

        # Teclas
        key_pressed = cv2.waitKey(1)
        if key_pressed & 0xFF == ord('q'):
            break
        elif key_pressed & 0xFF == ord('1'):
            ocr_mode = 'knn'
        elif key_pressed & 0xFF == ord('2'):
            ocr_mode = 'mlp'
        elif key_pressed & 0xFF == ord('3'):
            ocr_mode = 'tesseract'
        elif key_pressed & 0xFF == ord('4'):
            use_google_vision = True
        elif key_pressed & 0xFF == ord('m'):
            manual_board_contour = not manual_board_contour

    Config.CAMERA.release()
    cv2.destroyAllWindows()