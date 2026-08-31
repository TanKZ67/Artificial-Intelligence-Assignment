import os
import sys

import cv2
import joblib
import mediapipe as mp
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "Models", "asl_rf_model.pkl")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def run():
    """Run the Random Forest ASL pipeline.

    Returns:
        "menu" if the user pressed ESC (go back to the model-selection menu)
        "quit" if the user pressed Q (exit the whole program)
    """
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

    current_word = ""
    frame_counter = 0
    pred_label = "None"
    confidence = 0.0
    action = "menu"

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

                    # 每 2 帧计算一次，保证不卡顿且实时性极佳
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

            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            cv2.putText(frame, f"Prediction: {pred_label} ({confidence * 100:.1f}%)", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Word: {current_word}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, "[Space] append  [C] clear  [ESC] menu  [Q] quit", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Real-Time ASL Translation (Random Forest)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                action = "quit"
                break
            elif key == 27:  # ESC -> back to menu
                action = "menu"
                break
            elif key == ord(' '):
                if pred_label not in ["None", "nothing", "del", "space"]:
                    current_word += pred_label
                elif pred_label == "space":
                    current_word += " "
                elif pred_label == "del":
                    current_word = current_word[:-1]
            elif key in (ord('c'), ord('C')):
                current_word = ""
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
