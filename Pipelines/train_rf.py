import os
import sys
import cv2
import joblib
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(BASE_DIR, "dataset", "asl_alphabet_train", "asl_alphabet_train")
if not os.path.exists(dataset_path):
    dataset_path = os.path.join(BASE_DIR, "dataset", "asl_alphabet_train")

SAMPLES_PER_CLASS = 1500 
RANDOM_SEED = 42

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

X, y = [], []
labels = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
print(f"Random Forest Pipeline: Found {len(labels)} classes. Extracting landmarks...", flush=True)

for idx, label in enumerate(labels, 1):
    folder_path = os.path.join(dataset_path, label)
    images = sorted(os.listdir(folder_path))[:SAMPLES_PER_CLASS]
    extracted = 0

    for img_name in images:
        img = cv2.imread(os.path.join(folder_path, img_name))
        if img is None:
            continue

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_img)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
                
                f_orig, f_flip = [], []
                for lm in landmarks:
                    dx, dy, dz = lm.x - base_x, lm.y - base_y, lm.z - base_z
                    f_orig.extend([dx, dy, dz])
                    f_flip.extend([-dx, dy, dz])
                
                X.append(f_orig)
                y.append(label)
                X.append(f_flip)
                y.append(label)
                extracted += 2

    print(f"[{idx}/{len(labels)}] '{label}' processed ({extracted} samples)", flush=True)

hands.close()
X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

print("\n--- Training Random Forest (100 Trees) ---", flush=True)
rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, rf_pred)
print(f"\n✅ Random Forest Test Accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, rf_pred))

save_dir = os.path.join(BASE_DIR, "Models")
os.makedirs(save_dir, exist_ok=True)
joblib.dump(rf_model, os.path.join(save_dir, "asl_rf_model.pkl"), compress=3)
print("✅ Saved to Models/asl_rf_model.pkl")