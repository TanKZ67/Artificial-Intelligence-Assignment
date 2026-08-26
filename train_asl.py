import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Dataset path configuration
base_path = "dataset/asl_alphabet_train/asl_alphabet_train"
if not os.path.exists(base_path):
    base_path = "dataset/asl_alphabet_train"

# 2. Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

X = []
y = []

# Number of samples per class
SAMPLES_PER_CLASS = 800

labels = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
labels.sort()

print(f"Found {len(labels)} classes: {labels}")
print("Extracting 3D hand landmarks with data augmentation, please wait...")

for label in labels:
    folder_path = os.path.join(base_path, label)
    images = os.listdir(folder_path)[:SAMPLES_PER_CLASS]
    extracted = 0

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_img)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                # Wrist point (ID 0) as coordinate origin
                base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
                
                feature_orig = []
                feature_flip = []
                
                for lm in landmarks:
                    dx = lm.x - base_x
                    dy = lm.y - base_y
                    dz = lm.z - base_z
                    
                    # Original sample (63 features: x, y, z)
                    feature_orig.extend([dx, dy, dz])
                    # Horizontal flip augmentation for left/right hand support (-dx, dy, dz)
                    feature_flip.extend([-dx, dy, dz])
                
                # Append both original and augmented data
                X.append(feature_orig)
                y.append(label)
                X.append(feature_flip)
                y.append(label)
                extracted += 2

    print(f"Class [{label}]: Successfully processed {extracted} samples (including flip)")

hands.close()

X = np.array(X)
y = np.array(y)

print(f"\nFeature extraction completed! Total samples: {len(X)}, Feature dimension: {X.shape[1]}")

# 3. Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train Support Vector Machine (SVM)
print("\n--- Training Support Vector Machine (SVM) Model ---")
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# 5. Model Evaluation
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

print(f"\nSVM Test Accuracy: {svm_acc * 100:.2f}%")
print("\n--- Classification Report ---")
print(classification_report(y_test, svm_pred))

# 6. Save trained model
joblib.dump(svm_model, "asl_svm_model.pkl")
print("\nModel successfully saved as 'asl_svm_model.pkl'!")