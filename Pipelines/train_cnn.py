import json
import os
import random
import sys

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from Utils.hand_crop_utils import crop_and_mask_hand

MODELS_DIR = os.path.join(BASE_DIR, "Models")
os.makedirs(MODELS_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# 2. Configuration 
# --------------------------------------------------------------------------
IMAGE_SIZE = 96 
CACHE_SIZE = 96 
BATCH_SIZE = 32
EPOCHS_HEAD = 12 
EPOCHS_FINETUNE = 8 
FINE_TUNE_LAYERS = 40 
VALIDATION_SPLIT = 0.2
LABEL_SMOOTHING = 0.1 
RANDOM_SEED = 42
SAMPLES_PER_CLASS = 1500

MODEL_OUT_PATH = os.path.join(MODELS_DIR, "asl_cnn_model.h5")
TFLITE_OUT_PATH = os.path.join(MODELS_DIR, "asl_cnn_model.tflite") 
LABELS_OUT_PATH = os.path.join(MODELS_DIR, "asl_cnn_labels.txt")
HISTORY_JSON_PATH = os.path.join(MODELS_DIR, "training_history.json") 

random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

mp_hands = mp.solutions.hands


# --------------------------------------------------------------------------
# 3. Locating the raw dataset and listing samples
# --------------------------------------------------------------------------
def resolve_raw_dataset_dirs():
    single = os.path.join(BASE_DIR, "dataset", "asl_alphabet_train", "asl_alphabet_train")
    if not os.path.isdir(single):
        single = os.path.join(BASE_DIR, "dataset", "asl_alphabet_train")
    if not os.path.isdir(single):
        single = os.path.join(BASE_DIR, "dataset")
    return [single]


def list_samples(dataset_dirs, samples_per_class=SAMPLES_PER_CLASS):
    valid_dirs = [d for d in dataset_dirs if os.path.isdir(d)]
    if not valid_dirs:
        raise FileNotFoundError(
            f"Could not find a raw dataset under any of {dataset_dirs}.\n"
            "Download the Kaggle ASL alphabet dataset and extract it into the 'dataset' folder first."
        )

    filepaths, labels = [], []
    class_name_set = set()
    for raw_dir in valid_dirs:
        class_names = sorted(d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d)))
        for label in class_names:
            class_name_set.add(label)
            class_dir = os.path.join(raw_dir, label)
            filenames = sorted(os.listdir(class_dir))[:samples_per_class]
            for filename in filenames:
                filepaths.append(os.path.join(class_dir, filename))
                labels.append(label)

    class_names = sorted(class_name_set)
    print(f"Found {len(filepaths)} raw images across {len(class_names)} classes: {class_names}")
    return filepaths, labels, class_names


def split_train_val(filepaths, labels, validation_split=VALIDATION_SPLIT, seed=RANDOM_SEED):
    by_class = {}
    for path, label in zip(filepaths, labels):
        by_class.setdefault(label, []).append(path)

    rng = random.Random(seed)
    train_files, train_labels, val_files, val_labels = [], [], [], []
    for label, paths in by_class.items():
        shuffled = paths[:]
        rng.shuffle(shuffled)
        split_at = int(len(shuffled) * (1 - validation_split))
        for path in shuffled[:split_at]:
            train_files.append(path)
            train_labels.append(label)
        for path in shuffled[split_at:]:
            val_files.append(path)
            val_labels.append(label)

    return train_files, train_labels, val_files, val_labels


# --------------------------------------------------------------------------
# 4. Building the in-memory hand-crop cache
# --------------------------------------------------------------------------
def build_hand_crop_cache(filepaths, labels, class_to_index, cache_size=CACHE_SIZE, tag=""):
    images = np.zeros((len(filepaths), cache_size, cache_size, 3), dtype=np.uint8)
    label_indices = np.zeros(len(filepaths), dtype=np.int64)
    detected_count = 0

    hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    try:
        for i, (path, label) in enumerate(zip(filepaths, labels)):
            if i > 0 and i % 500 == 0:
                hands_detector.close()
                hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

            image = cv2.imread(path)
            if image is None:
                continue
            crop, hand_found, _box, _landmarks = crop_and_mask_hand(image, hands_detector)
            if crop is None:
                continue
            images[i] = cv2.resize(crop, (cache_size, cache_size))
            label_indices[i] = class_to_index[label]
            detected_count += int(hand_found)

            if (i + 1) % 500 == 0 or (i + 1) == len(filepaths):
                print(f"  [{tag}] processed {i + 1}/{len(filepaths)} images (hand detected in {detected_count})", flush=True)
    finally:
        hands_detector.close()

    return images, label_indices


# --------------------------------------------------------------------------
# 5. Augmentation - applied on top of the cached crops
# --------------------------------------------------------------------------
def add_random_noise(image):
    if random.random() < 0.5:
        return image
    std = random.uniform(2.0, 12.0)
    noise = np.random.normal(0.0, std, image.shape).astype("float32")
    return np.clip(image.astype("float32") + noise, 0, 255).astype("uint8")


def add_random_blur(image):
    if random.random() < 0.6:
        return image
    ksize = random.choice([3, 5])
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def add_random_cutout(image, max_patches=2, max_frac=0.18):
    if random.random() < 0.5:
        return image
    h, w = image.shape[:2]
    output = image.copy()
    for _ in range(random.randint(1, max_patches)):
        patch_h = max(int(h * random.uniform(0.08, max_frac)), 1)
        patch_w = max(int(w * random.uniform(0.08, max_frac)), 1)
        y0 = random.randint(0, max(h - patch_h, 0))
        x0 = random.randint(0, max(w - patch_w, 0))
        output[y0 : y0 + patch_h, x0 : x0 + patch_w] = np.random.randint(
            0, 255, size=(patch_h, patch_w, image.shape[2]), dtype="uint8"
        )
    return output


def random_affine_augment(image, rotation_range=15, shift_range=0.1, zoom_range=0.2, shear_range=0.1):
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    angle = random.uniform(-rotation_range, rotation_range)
    scale = 1.0 + random.uniform(-zoom_range, zoom_range)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)

    shear = random.uniform(-shear_range, shear_range)
    matrix[0, 1] += shear
    matrix[0, 2] += random.uniform(-shift_range, shift_range) * w
    matrix[1, 2] += random.uniform(-shift_range, shift_range) * h

    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def random_brightness(image, brightness_range=(0.6, 1.4)):
    factor = random.uniform(*brightness_range)
    return np.clip(image.astype("float32") * factor, 0, 255).astype("uint8")


def random_channel_shift(image, intensity=30.0):
    shift = np.random.uniform(-intensity, intensity, size=(1, 1, 3)).astype("float32")
    return np.clip(image.astype("float32") + shift, 0, 255).astype("uint8")


def random_horizontal_flip(image, p=0.5):
    if random.random() < p:
        return cv2.flip(image, 1)
    return image


def train_time_augment(image):
    image = random_horizontal_flip(image)
    image = random_affine_augment(image)
    image = random_brightness(image)
    image = random_channel_shift(image)
    image = add_random_noise(image)
    image = add_random_blur(image)
    image = add_random_cutout(image)
    return image


def finalize_for_model(image_bgr, image_size=IMAGE_SIZE):
    resized = cv2.resize(image_bgr, (image_size, image_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32")
    return preprocess_input(rgb)


# --------------------------------------------------------------------------
# 6. Cached Hand Crop Sequence
# --------------------------------------------------------------------------
class CachedHandCropSequence(tf.keras.utils.Sequence):
    def __init__(self, images, label_indices, num_classes, batch_size=BATCH_SIZE,
                 image_size=IMAGE_SIZE, augment=False, shuffle=False):
        super().__init__()
        self.images = images
        self.label_indices = label_indices
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(images))
        self.on_epoch_end()

    def __len__(self):
        return max(1, int(np.ceil(len(self.images) / self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = np.zeros((len(batch_indices), self.image_size, self.image_size, 3), dtype="float32")
        batch_y = np.zeros((len(batch_indices), self.num_classes), dtype="float32")

        for i, sample_idx in enumerate(batch_indices):
            image = self.images[sample_idx]
            if self.augment:
                image = train_time_augment(image)
            batch_x[i] = finalize_for_model(image, self.image_size)
            batch_y[i, self.label_indices[sample_idx]] = 1.0

        return batch_x, batch_y


# --------------------------------------------------------------------------
# 7. Model Architecture Definition
# --------------------------------------------------------------------------
def build_model(input_shape, num_classes):
    try:
        base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    except Exception as exc:
        print(f"Could not download ImageNet weights ({exc}); falling back to random initialisation.")
        base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights=None)

    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="asl_cnn_mobilenetv2")
    return model, base_model


def save_label_map(class_to_index, path):
    index_to_label = {v: k for k, v in class_to_index.items()}
    with open(path, "w") as f:
        for idx in sorted(index_to_label):
            f.write(f"{idx},{index_to_label[idx]}\n")


def export_tflite(model, path):
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(path, "wb") as f:
            f.write(tflite_model)
        print(f"Exported TensorFlow Lite model to '{path}' ({len(tflite_model) / 1024:.0f} KB).")
    except Exception as exc:
        print(f"Warning: TFLite conversion failed ({exc}).")


def save_history_json(history_head, history_finetune, path):
    combined = {
        "phase_boundary": len(history_head.epoch),
        "accuracy": history_head.history["accuracy"] + history_finetune.history["accuracy"],
        "val_accuracy": history_head.history["val_accuracy"] + history_finetune.history["val_accuracy"],
        "loss": history_head.history["loss"] + history_finetune.history["loss"],
        "val_loss": history_head.history["val_loss"] + history_finetune.history["val_loss"],
    }
    with open(path, "w") as f:
        json.dump(combined, f)
    print(f"Saved training history to '{path}'.")


# --------------------------------------------------------------------------
# 8. Training Entry Point
# --------------------------------------------------------------------------
def main():
    dataset_dirs = resolve_raw_dataset_dirs()
    filepaths, labels, class_names = list_samples(dataset_dirs)
    class_to_index = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    train_files, train_labels, val_files, val_labels = split_train_val(filepaths, labels)
    print(f"Train/validation split: {len(train_files)} train, {len(val_files)} validation images.")

    print("\nBuilding the in-memory hand-crop cache (MediaPipe runs once per image here)...")
    train_images, train_label_indices = build_hand_crop_cache(train_files, train_labels, class_to_index, tag="train")
    val_images, val_label_indices = build_hand_crop_cache(val_files, val_labels, class_to_index, tag="val")

    train_gen = CachedHandCropSequence(
        train_images, train_label_indices, num_classes, augment=True, shuffle=True
    )
    val_gen = CachedHandCropSequence(
        val_images, val_label_indices, num_classes, augment=False, shuffle=False
    )

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)

    model, base_model = build_model((IMAGE_SIZE, IMAGE_SIZE, 3), num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn, metrics=["accuracy"])
    model.summary()

    head_callbacks = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        callbacks.ModelCheckpoint(MODEL_OUT_PATH, monitor="val_accuracy", save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    print("\n--- Phase 1: training the classification head (backbone frozen) ---")
    history_head = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_HEAD,
        callbacks=head_callbacks,
    )

    print("\n--- Phase 2: fine-tuning the top of the backbone ---")
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=loss_fn, metrics=["accuracy"])

    finetune_callbacks = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        callbacks.ModelCheckpoint(MODEL_OUT_PATH, monitor="val_accuracy", save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]

    last_head_epoch = history_head.epoch[-1] if history_head.epoch else -1
    history_finetune = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_HEAD + EPOCHS_FINETUNE,
        initial_epoch=last_head_epoch + 1,
        callbacks=finetune_callbacks,
    )

    model.save(MODEL_OUT_PATH)
    save_label_map(class_to_index, LABELS_OUT_PATH)
    export_tflite(model, TFLITE_OUT_PATH)
    save_history_json(history_head, history_finetune, HISTORY_JSON_PATH)

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"\nValidation accuracy: {val_acc * 100:.2f}% (loss={val_loss:.4f})")
    print(f"Model saved to '{MODEL_OUT_PATH}' / '{TFLITE_OUT_PATH}', labels saved to '{LABELS_OUT_PATH}'.")


if __name__ == "__main__":
    main()