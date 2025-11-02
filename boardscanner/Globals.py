import threading
import mediapipe as mp

class Globals:
    # Fase 1
    original_frame = None
    hsv_frame = None
    hsv_mask_frame = None
    canny_frame = None
    closure_frame = None
    contour_frame = None
    board_frame = None

    # Fase 2
    board_adaptive_threshold_frame = None
    board_dilated_frame = None
    board_with_word_contours_frame = None

    # EMNIST
    emnist_mlp_frame = None
    emnist_knn_frame = None
    # Google Vision API
    google_vision_frame = None

    lock = threading.Lock()
    emnist_model = None
    emnist_mappings = None
    emnist_knn_model = None
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils