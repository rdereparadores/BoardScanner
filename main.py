import threading
import time
import cv2
from flask import Flask
from flask import render_template
from flask import Response

from boardscanner.Globals import Globals
from boardscanner.main_loop import main_loop
app = Flask(__name__)

def generate_encoded_frame(frame_name):
    while True:
        with Globals.lock:
            if frame_name == 'original':
                frame = Globals.original_frame
            elif frame_name == 'hsv':
                frame = Globals.hsv_frame
            elif frame_name == 'hsv_mask':
                frame = Globals.hsv_mask_frame
            elif frame_name == 'canny':
                frame = Globals.canny_frame
            elif frame_name == 'closure':
                frame = Globals.closure_frame
            elif frame_name == 'contour':
                frame = Globals.contour_frame
            elif frame_name == 'board':
                frame = Globals.board_frame
            elif frame_name == 'board_adaptive_threshold':
                frame = Globals.board_adaptive_threshold_frame
            elif frame_name == 'board_dilated':
                frame = Globals.board_dilated_frame
            elif frame_name == 'board_with_word_contours':
                frame = Globals.board_with_word_contours_frame
            elif frame_name == 'emnist_knn':
                frame = Globals.emnist_knn_frame
            elif frame_name == 'emnist_mlp':
                frame = Globals.emnist_mlp_frame
            elif frame_name == 'google_vision':
                frame = Globals.google_vision_frame
            else:
                frame = Globals.original_frame

        _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded) + b'\r\n'
        time.sleep(0.1) # 10 FPS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/original_frame')
def original_frame():
    return Response(generate_encoded_frame('original'), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/hsv_frame')
def hsv_frame():
    return Response(generate_encoded_frame('hsv'), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/hsv_mask_frame')
def hsv_mask_frame():
    return Response(generate_encoded_frame('hsv_mask'), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/canny_frame')
def canny_frame():
    return Response(generate_encoded_frame('canny'), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/closure_frame')
def closure_frame():
    return Response(generate_encoded_frame('closure'), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/contour_frame')
def contour_frame():
    return Response(generate_encoded_frame('contour'), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/board_frame')
def board_frame():
    return Response(generate_encoded_frame('board'), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/board_adaptive_threshold_frame')
def board_adaptive_threshold_frame():
    return Response(generate_encoded_frame('board_adaptive_threshold'), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/board_dilated_frame')
def board_dilated_frame():
    return Response(generate_encoded_frame('board_dilated'), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/board_with_word_contours_frame')
def board_with_word_contours_frame():
    return Response(generate_encoded_frame('board_with_word_contours'), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/emnist_knn_frame')
def emnist_knn_frame():
    return Response(generate_encoded_frame('emnist_knn'), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/emnist_mlp_frame')
def emnist_mlp_frame():
    return Response(generate_encoded_frame('emnist_mlp'), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/google_vision_frame')
def google_vision_frame():
    return Response(generate_encoded_frame('google_vision'), mimetype="multipart/x-mixed-replace; boundary=frame")

def flask_loop():
    app.run(debug=False, port=8080)

if __name__ == '__main__':
    t = threading.Thread(target=flask_loop, daemon=True)
    t.start()

    main_loop()
