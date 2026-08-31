import os
import sys
import csv
import random

import cv2
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")
PIPELINES_DIR = os.path.join(PROJECT_ROOT, "Pipelines")

sys.path.append(PROJECT_ROOT)
sys.path.append(PIPELINES_DIR)

DATASET_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "dataset", "asl_alphabet_train", "asl_alphabet_train"),
    os.path.join(PROJECT_ROOT, "dataset", "asl_alphabet_train"),
    r"C:\Users\Ming\Downloads\archive (4)\asl_alphabet_train\asl_alphabet_train",
]

SVM_MODEL_PATH = os.path.join(MODELS_DIR, "asl_svm_model.pkl")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "asl_rf_model.pkl")
CNN_TFLITE_PATH = os.path.join(MODELS_DIR, "asl_cnn_model.tflite")
CNN_LABELS_PATH = os.path.join(MODELS_DIR, "asl_cnn_labels.txt")

SAMPLES_PER_CLASS = 700 
TEST_SIZE = 0.5
RANDOM_SEED = 42

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TOP_CONFUSIONS_TO_SHOW = 10 

SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue",
    ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#104281", "#0d366b"],
)

def resolve_dataset_dir():
    for path in DATASET_CANDIDATES:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        "Could not find the ASL Alphabet dataset in any of:\n  "
        + "\n  ".join(DATASET_CANDIDATES)
        + "\nEdit DATASET_CANDIDATES at the top of this script and add the "
        "correct path to your extracted 'asl_alphabet_train' folder."
    )


def sample_image_paths(dataset_dir, samples_per_class=SAMPLES_PER_CLASS, seed=RANDOM_SEED):
    rng = random.Random(seed)
    class_names = sorted(d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)))

    filepaths, labels = [], []
    for label in class_names:
        class_dir = os.path.join(dataset_dir, label)
        filenames = sorted(os.listdir(class_dir))
        if len(filenames) > samples_per_class:
            filenames = rng.sample(filenames, samples_per_class)
        for filename in filenames:
            filepaths.append(os.path.join(class_dir, filename))
            labels.append(label)
    return filepaths, labels, class_names


def split_test_only(filepaths, labels, test_size=TEST_SIZE, seed=RANDOM_SEED):
    _, test_files, _, test_labels = train_test_split(
        filepaths, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    return test_files, test_labels

def extract_landmark_features(filepaths, hands_detector):
    features, valid_idx = [], []
    for i, path in enumerate(filepaths):
        img = cv2.imread(path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands_detector.process(rgb)
        if not result.multi_hand_landmarks:
            continue

        landmarks = result.multi_hand_landmarks[0].landmark
        base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
        feature = []
        for lm in landmarks:
            feature.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])

        features.append(feature)
        valid_idx.append(i)

        if (i + 1) % 200 == 0 or (i + 1) == len(filepaths):
            print(f"    landmark extraction: {i + 1}/{len(filepaths)} images processed", flush=True)

    return np.array(features), valid_idx


def evaluate_sklearn_model(name, slug, model_path, X_test, y_true):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    if not os.path.exists(model_path):
        print(f"  Model file not found at '{model_path}' - skipping.")
        return None
    if len(X_test) == 0:
        print("  No usable test samples (no hands detected) - skipping.")
        return None

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    return report_metrics(y_true, y_pred, name, slug)

def evaluate_cnn_model(test_files, test_labels):
    print(f"\n{'=' * 60}\nCNN (MobileNetV2 + MediaPipe crop)\n{'=' * 60}")

    if not (os.path.exists(CNN_TFLITE_PATH) and os.path.exists(CNN_LABELS_PATH)):
        print(f"  Model/labels not found under '{MODELS_DIR}' - skipping.")
        return None

    import mediapipe as mp
    from main_cnn import HandSignInterpreter, predict_averaged_probabilities

    interpreter = HandSignInterpreter(model_path=CNN_TFLITE_PATH, labels_path=CNN_LABELS_PATH)
    mp_hands = mp.solutions.hands

    y_true, y_pred = [], []
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands_detector:
        for i, (path, true_label) in enumerate(zip(test_files, test_labels)):
            img = cv2.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands_detector.process(rgb)

            probs = None
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                probs = predict_averaged_probabilities(img, hand_landmarks, w, h, interpreter)

            pred_label = "nothing" if probs is None else interpreter.label_for_index(int(np.argmax(probs)))

            y_true.append(true_label)
            y_pred.append(pred_label)

            if (i + 1) % 200 == 0 or (i + 1) == len(test_files):
                print(f"    CNN inference: {i + 1}/{len(test_files)} images processed", flush=True)

    return report_metrics(y_true, y_pred, "CNN (MobileNetV2 + MediaPipe crop)", "CNN")

def save_confusion_matrix(y_true, y_pred, model_name, slug):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    csv_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{slug}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Actual \\ Predicted"] + labels)
        for label, row in zip(labels, cm):
            writer.writerow([label] + list(row))

    png_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{slug}.png")
    side = max(6.0, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(cm, cmap=SEQUENTIAL_BLUE, aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(f"Confusion Matrix - {model_name}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Number of samples")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print("\n  Confusion matrix saved to:")
    print(f"    {csv_path}")
    print(f"    {png_path}")

    confusions = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i, j] > 0:
                confusions.append((int(cm[i, j]), true_label, pred_label))
    confusions.sort(reverse=True)

    if confusions:
        n_show = min(TOP_CONFUSIONS_TO_SHOW, len(confusions))
        print(f"\n  Top {n_show} most-confused label pairs (actual -> predicted):")
        for count, true_label, pred_label in confusions[:n_show]:
            print(f"    {true_label:>10} -> {pred_label:<10} : {count} times")

def report_metrics(y_true, y_pred, model_name, slug):
    acc = accuracy_score(y_true, y_pred)
    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec_m = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_m = recall_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n  Samples evaluated   : {len(y_true)}")
    print(f"  Accuracy            : {acc * 100:.2f}%")
    print(f"  Precision (weighted): {prec_w * 100:.2f}%   (macro: {prec_m * 100:.2f}%)")
    print(f"  Recall    (weighted): {rec_w * 100:.2f}%   (macro: {rec_m * 100:.2f}%)")
    print(f"  F1-score  (weighted): {f1_w * 100:.2f}%")
    print("\n  Per-class report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    save_confusion_matrix(y_true, y_pred, model_name, slug)

    return {"accuracy": acc, "precision_weighted": prec_w, "recall_weighted": rec_w, "f1_weighted": f1_w}

def main():
    dataset_dir = resolve_dataset_dir()
    print(f"Using dataset       : {dataset_dir}")
    print(f"Sampling            : up to {SAMPLES_PER_CLASS} images/class, "
          f"then a {int(TEST_SIZE * 100)}% held-out test split (seed={RANDOM_SEED})")

    filepaths, labels, class_names = sample_image_paths(dataset_dir)
    print(f"Sampled             : {len(filepaths)} images across {len(class_names)} classes")

    test_files, test_labels = split_test_only(filepaths, labels)
    print(f"Held-out test set   : {len(test_files)} images\n")

    print("Extracting MediaPipe hand-landmark features (shared by SVM and Random Forest)...")
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hd:
        X_test, valid_idx = extract_landmark_features(test_files, hd)
    y_true_landmark = [test_labels[i] for i in valid_idx]
    skipped = len(test_files) - len(valid_idx)
    if skipped:
        print(f"  Note: {skipped}/{len(test_files)} images had no hand detected by MediaPipe "
              "and were excluded from the SVM/Random Forest evaluation.")

    results = {}
    for name, path, fn in [
        ("SVM", SVM_MODEL_PATH, lambda: evaluate_sklearn_model("Support Vector Machine (SVM)", "SVM", SVM_MODEL_PATH, X_test, y_true_landmark)),
        ("Random Forest", RF_MODEL_PATH, lambda: evaluate_sklearn_model("Random Forest", "RandomForest", RF_MODEL_PATH, X_test, y_true_landmark)),
        ("CNN", None, lambda: evaluate_cnn_model(test_files, test_labels)),
    ]:
        try:
            results[name] = fn()
        except Exception as exc:
            print(f"\n  {name} evaluation raised an error and was skipped: {exc}")
            results[name] = None

    print(f"\n{'=' * 60}\nSummary\n{'=' * 60}")
    header = f"{'Model':<20}{'Accuracy':>12}{'Precision':>14}{'Recall':>12}{'F1':>10}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        if m is None:
            print(f"{name:<20}{'skipped':>12}")
            continue
        print(
            f"{name:<20}{m['accuracy'] * 100:>11.2f}%"
            f"{m['precision_weighted'] * 100:>13.2f}%"
            f"{m['recall_weighted'] * 100:>11.2f}%"
            f"{m['f1_weighted'] * 100:>9.2f}%"
        )
    print(
        "\n(Precision/Recall/F1 above are weighted averages across all classes; "
        "see each model's per-class report further up for individual letters.)"
    )
    print(f"Confusion-matrix CSVs and heatmap PNGs were saved under: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
