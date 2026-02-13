import cv2
import math
import numpy as np


def hough_lines_transform(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []
    for [[x1, y1, x2, y2]] in lines:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    img_rotated = cv2.warpAffine(image, M, (w, h))
    return img_rotated


def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_clahe(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def reduce_highlights(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Уменьшаем только очень яркие области
    mask = v > 220
    v = np.where(mask, (v * 0.85).astype(np.uint8), v)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def preprocess_document_image(image):
    # 1. Гамма-коррекция (если тёмное)
    # if np.mean(image) < 90:
    #     image = adjust_gamma(image, gamma=1.4)
    #
    # 2. CLAHE — улучшение контраста
    # image = apply_clahe(image)

    # 3. Уменьшение бликов
    # image = reduce_highlights(image)

    # 4. Медианный фильтр — убрать шум
    # image = cv2.medianBlur(image, 3)

    # 5. Повышение резкости
    # image = sharpen_image(image)

    image = hough_lines_transform(image)

    return image
