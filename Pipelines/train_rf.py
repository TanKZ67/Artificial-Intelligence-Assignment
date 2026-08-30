import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

base_path = r"C:\Users\tanka\Downloads\archive (2)\asl_alphabet_train\asl_alphabet_train"
if not os.path.exists(base_path):
    base_path = r"C:\Users\tanka\Downloads\archive (2)\asl_alphabet_train"

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

X, y = [], []
SAMPLES_PER_CLASS = 1500  # 每类抽取数量

labels = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
labels.sort()

total_classes = len(labels)
print(f"找到 {total_classes} 个类别，开始训练你的 Random Forest 模型...\n", flush=True)

# 遍历每一个类别文件夹
for idx, label in enumerate(labels, 1):
    folder_path = os.path.join(base_path, label)
    images = os.listdir(folder_path)[:SAMPLES_PER_CLASS]
    extracted_count = 0

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
                base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
                
                feature_orig, feature_flip = [], []
                for lm in landmarks:
                    dx, dy, dz = lm.x - base_x, lm.y - base_y, lm.z - base_z
                    feature_orig.extend([dx, dy, dz])
                    feature_flip.extend([-dx, dy, dz]) # 左右手镜像翻转增强
                
                X.append(feature_orig)
                y.append(label)
                X.append(feature_flip)
                y.append(label)
                extracted_count += 2

    # 关键：加上这行打印，终端就会实时显示进度！
    print(f"[{idx}/{total_classes}] 类别 '{label}' 处理完成，共提取 {extracted_count} 条数据", flush=True)

hands.close()
X, y = np.array(X), np.array(y)

print("\n--- 正在划分数据集 ---", flush=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("--- 开始训练 Random Forest 算法（约需 10-20 秒）---", flush=True)
rf_model = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, rf_pred)
print(f"\n✅ 训练完成！Random Forest 模型准确率: {acc * 100:.2f}%", flush=True)

joblib.dump(rf_model, "asl_rf_model.pkl")
print("✅ 模型已成功保存为 'asl_rf_model.pkl'！", flush=True)