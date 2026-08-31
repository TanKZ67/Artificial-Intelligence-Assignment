import os
import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from Utils.hand_crop_utils import BOX_PADDING_RATIO, apply_hand_mask, hand_bounding_box, letterbox_square

MODEL_PATH = os.path.join(BASE_DIR, "Models", "asl_cnn_model.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "Models", "asl_cnn_labels.txt")
IMAGE_SIZE = 96 

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
PROCESS_EVERY_N_FRAMES = 2 

CONFIDENCE_THRESHOLD = 0.75
AUTO_ENTER_THRESHOLD = 0.90   
COOLDOWN_SAME_CHAR = 2.0      

SMOOTHING_WINDOW = 5 
TTA_PADDING_OFFSETS = (-0.08, 0.0, 0.08) 
SPECIAL_TOKENS = {"space", "del", "nothing"}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class FrameGrabber:

    def __init__(self, capture):
        import threading

        self.capture = capture
        self._lock = threading.Lock()
        self._latest_frame = None
        self._ok = True
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stopped:
            ok, frame = self.capture.read()
            with self._lock:
                self._ok = ok
                if ok:
                    self._latest_frame = frame
            if not ok:
                break

    def read(self):
        with self._lock:
            if self._latest_frame is None:
                return self._ok, None
            return self._ok, self._latest_frame.copy()

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=1.0)


class HandSignInterpreter:

    def __init__(self, model_path=MODEL_PATH, labels_path=LABELS_PATH, image_size=IMAGE_SIZE):
        self.image_size = image_size
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at '{model_path}'. Check if Models/ folder has the tflite file.")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found at '{labels_path}'. Check if Models/ folder has the txt file.")
            
        print(f"Loading TFLite model from '{model_path}'...")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input_detail = self.interpreter.get_input_details()[0]
        self._output_detail = self.interpreter.get_output_details()[0]
        self.index_to_label = self._load_labels(labels_path)

    @staticmethod
    def _load_labels(path):
        mapping = {}
        with open(path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                idx_str, name = line.split(",", 1)
                mapping[int(idx_str)] = name
        return mapping

    def _preprocess(self, masked_square_crop):
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        resized = cv2.resize(masked_square_crop, (self.image_size, self.image_size)).astype("float32")
        rgb = cv2.cvtColor(resized.astype("uint8"), cv2.COLOR_BGR2RGB).astype("float32")
        normalized = preprocess_input(rgb) 
        batch = np.expand_dims(normalized, axis=0)
        return batch.astype(self._input_detail["dtype"])

    def predict_probabilities(self, masked_square_crop):
        batch = self._preprocess(masked_square_crop)
        self.interpreter.set_tensor(self._input_detail["index"], batch)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self._output_detail["index"])[0]

    def label_for_index(self, index):
        return self.index_to_label.get(index, "?")


def predict_averaged_probabilities(frame, hand_landmarks, frame_w, frame_h, interpreter):
    probabilities_list = []
    for offset in TTA_PADDING_OFFSETS:
        padding_ratio = max(BOX_PADDING_RATIO + offset, 0.05)
        x1, y1, x2, y2 = hand_bounding_box(hand_landmarks.landmark, frame_w, frame_h, padding_ratio)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        masked = apply_hand_mask(crop, hand_landmarks.landmark, x1, y1, frame_w, frame_h)
        squared = letterbox_square(masked)
        probabilities_list.append(interpreter.predict_probabilities(squared))

    if not probabilities_list:
        return None
    return np.mean(probabilities_list, axis=0)


class RollingPrediction:

    def __init__(self, window_size=SMOOTHING_WINDOW):
        self.buffer = deque(maxlen=window_size)

    def reset(self):
        self.buffer.clear()

    def update(self, probabilities):
        self.buffer.append(probabilities)
        averaged = np.mean(self.buffer, axis=0)
        best_index = int(np.argmax(averaged))
        confidence = float(averaged[best_index])
        return best_index, confidence


class WordBuilder:

    def __init__(self):
        self.text = ""

    def append_letter(self, label):
        if label == "space":
            self.text += " "
        elif label == "del":
            self.text = self.text[:-1]
        elif label not in SPECIAL_TOKENS and label != "None":
            self.text += label

    def backspace(self):
        self.text = self.text[:-1]

    def clear(self):
        self.text = ""


def draw_overlay(frame, box, label, confidence, word):
    if box is not None:
        x1, y1, x2, y2 = box
        box_color = (0, 200, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 140, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    else:
        box_color = (0, 0, 255)

    cv2.putText(
        frame,
        f"Sign: {label}  ({confidence * 100:.1f}%)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        box_color,
        2,
    )
    cv2.putText(frame, f"Word: {word}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.putText(
        frame,
        "[Space] append  [Backspace] delete  [C] clear  [ESC] menu  [Q] quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )


def run():
    try:
        interpreter = HandSignInterpreter()
    except FileNotFoundError as exc:
        print(f"[CNN] {exc}")
        return "menu"

    word_builder = WordBuilder()
    smoother = RollingPrediction()

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("[CNN] Could not open webcam (index 0).")
        return "menu"
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    grabber = FrameGrabber(capture)

    window_name = "CNN ASL Word Builder"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CAPTURE_WIDTH, CAPTURE_HEIGHT)

    frame_index = 0
    last_box, last_label, last_confidence, last_landmarks = None, "None", 0.0, None
    action = "menu"

   
    last_inserted_label = None
    last_inserted_time = 0.0

    print("CNN + MediaPipe ASL word builder started (TFLite runtime).")
    print("[Space] append  [Backspace] delete  [C] clear  [ESC] back to menu  [Q] quit")

    try:
        with mp_hands.Hands(
            max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
        ) as hands_detector:
            while True:
                ok, frame = grabber.read()
                if not ok:
                    break
                if frame is None:
                    continue 

                frame = cv2.flip(frame, 1)
                frame_h, frame_w = frame.shape[:2]
                frame_index += 1

                if frame_index % PROCESS_EVERY_N_FRAMES == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands_detector.process(rgb_frame)

                    if result.multi_hand_landmarks:
                        hand_landmarks = result.multi_hand_landmarks[0]
                        box = hand_bounding_box(hand_landmarks.landmark, frame_w, frame_h, BOX_PADDING_RATIO)
                        probabilities = predict_averaged_probabilities(frame, hand_landmarks, frame_w, frame_h, interpreter)
                        if probabilities is not None:
                            best_index, confidence = smoother.update(probabilities)
                            label = interpreter.label_for_index(best_index)
                            last_box, last_label, last_confidence = box, label, confidence
                            last_landmarks = hand_landmarks

                            
                            current_time = time.time()
                            if confidence >= AUTO_ENTER_THRESHOLD and label not in ("None", "nothing"):
                                if label != last_inserted_label:
                                    
                                    word_builder.append_letter(label)
                                    last_inserted_label = label
                                    last_inserted_time = current_time
                                else:
                                    
                                    if current_time - last_inserted_time >= COOLDOWN_SAME_CHAR:
                                        word_builder.append_letter(label)
                                        last_inserted_time = current_time
                        else:
                            last_box, last_label, last_confidence, last_landmarks = None, "None", 0.0, None
                    else:
                        smoother.reset() 
                        last_box, last_label, last_confidence, last_landmarks = None, "None", 0.0, None

                if last_landmarks is not None:
                    mp_draw.draw_landmarks(frame, last_landmarks, mp_hands.HAND_CONNECTIONS)

                draw_overlay(frame, last_box, last_label, last_confidence, word_builder.text)
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    action = "quit"
                    break
                elif key == 27:  # ESC -> back to menu
                    action = "menu"
                    break
                elif key == ord(" "):
                    word_builder.append_letter(last_label)
                    last_inserted_label = last_label
                    last_inserted_time = time.time()
                elif key in (8, 127): 
                    word_builder.backspace()
                elif key in (ord("c"), ord("C")):
                    word_builder.clear()
                    last_inserted_label = None
    finally:
        grabber.stop()
        capture.release()
        cv2.destroyAllWindows()
        print(f"Final word: {word_builder.text}")

    return action


def main():
    run()


if __name__ == "__main__":
    main()