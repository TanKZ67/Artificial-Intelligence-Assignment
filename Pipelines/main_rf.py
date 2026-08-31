import os
import sys
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "Models", "asl_rf_model.pkl")

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

AUTO_ENTER_THRESHOLD = 0.90   
COOLDOWN_SAME_CHAR = 2.0      

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def append_character(current_word, label):
    if label not in ["None", "nothing", "del", "space"]:
        current_word += label
    elif label == "space":
        current_word += " "
    elif label == "del":
        current_word = current_word[:-1]
    return current_word


def run():
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"[RF] Model file not found at '{MODEL_PATH}'. Check if Models/ folder has the pkl file.")
        return "menu"

    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[RF] Could not open webcam (index 0).")
        hands.close()
        return "menu"

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    window_name = "Real-Time ASL Translation (Random Forest)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CAPTURE_WIDTH, CAPTURE_HEIGHT)

    current_word = ""
    frame_counter = 0
    pred_label = "None"
    confidence = 0.0
    action = "menu"

    
    last_inserted_label = None
    last_inserted_time = 0.0

    print("Real-time ASL System Started (Random Forest)!")
    print("[Space] append  [C] clear  [ESC] back to menu  [Q] quit")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    if frame_counter % 2 == 0:
                        landmarks = hand_landmarks.landmark
                        base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
                        feature = []
                        for lm in landmarks:
                            feature.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])

                        probs = model.predict_proba([feature])[0]
                        max_idx = np.argmax(probs)
                        confidence = probs[max_idx]
                        pred_label = model.classes_[max_idx]

                        
                        current_time = time.time()
                        if confidence >= AUTO_ENTER_THRESHOLD and pred_label not in ["None", "nothing"]:
                            if pred_label != last_inserted_label:
                                current_word = append_character(current_word, pred_label)
                                last_inserted_label = pred_label
                                last_inserted_time = current_time
                            else:
                                if current_time - last_inserted_time >= COOLDOWN_SAME_CHAR:
                                    current_word = append_character(current_word, pred_label)
                                    last_inserted_time = current_time

            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            cv2.putText(frame, f"Prediction: {pred_label} ({confidence * 100:.1f}%)", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Word: {current_word}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            cv2.putText(frame, "[Space] append  [C] clear  [ESC] menu  [Q] quit", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                action = "quit"
                break
            elif key == 27:  # ESC -> back to menu
                action = "menu"
                break
            elif key == ord(' '):
                current_word = append_character(current_word, pred_label)
                last_inserted_label = pred_label
                last_inserted_time = time.time()
            elif key in (ord('c'), ord('C')):
                current_word = ""
                last_inserted_label = None
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
        print(f"Final word: {current_word}")

    return action


def main():
    run()


if __name__ == "__main__":
    main()