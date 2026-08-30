import cv2
import numpy as np

BOX_PADDING_RATIO = 0.35 
MASK_DILATE_RATIO = 0.14 
MASK_FILL_COLOR = (128, 128, 128) 


def hand_bounding_box(landmarks, frame_w, frame_h, padding_ratio=BOX_PADDING_RATIO):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min, x_max = min(xs) * frame_w, max(xs) * frame_w
    y_min, y_max = min(ys) * frame_h, max(ys) * frame_h

    box_w, box_h = x_max - x_min, y_max - y_min
    pad_x, pad_y = box_w * padding_ratio, box_h * padding_ratio

    x1 = max(int(x_min - pad_x), 0)
    y1 = max(int(y_min - pad_y), 0)
    x2 = min(int(x_max + pad_x), frame_w)
    y2 = min(int(y_max + pad_y), frame_h)
    return x1, y1, x2, y2


def apply_hand_mask(crop_bgr, landmarks, x1, y1, frame_w, frame_h, dilate_ratio=MASK_DILATE_RATIO,
                     fill_color=MASK_FILL_COLOR):
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return crop_bgr

    points = np.array(
        [[lm.x * frame_w - x1, lm.y * frame_h - y1] for lm in landmarks],
        dtype=np.float32,
    )
    hull = cv2.convexHull(points).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    kernel_size = max(int(max(h, w) * dilate_ratio), 1)
    kernel_size += 1 - (kernel_size % 2)  # cv2 wants an odd kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.dilate(mask, kernel)

    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=kernel_size / 3).astype(np.float32) / 255.0
    mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    background = np.full_like(crop_bgr, fill_color, dtype=np.float32)
    blended = crop_bgr.astype(np.float32) * mask_3ch + background * (1.0 - mask_3ch)
    return np.clip(blended, 0, 255).astype(np.uint8)


def letterbox_square(image):
    h, w = image.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)


def center_square_crop(image):
    h, w = image.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return image[y0 : y0 + side, x0 : x0 + side]


def crop_and_mask_hand(image_bgr, hands_detector, padding_ratio=BOX_PADDING_RATIO):
    h, w = image_bgr.shape[:2]
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb_image)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0].landmark
        box = hand_bounding_box(landmarks, w, h, padding_ratio)
        x1, y1, x2, y2 = box
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None, False, None, None
        crop = apply_hand_mask(crop, landmarks, x1, y1, w, h)
        crop = letterbox_square(crop)
        return crop, True, box, landmarks

    crop = center_square_crop(image_bgr)
    if crop.size == 0:
        return None, False, None, None
    return letterbox_square(crop), False, None, None
