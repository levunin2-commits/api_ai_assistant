import gradio as gr
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from src.preprocess_image import preprocess_document_image
from ultralytics import YOLO
from torchvision import transforms
from my_mobilenet import MyMobileNetV2
import easyocr

# ========== 1. Загрузка моделей ==========

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Классификатор
cls_model = MyMobileNetV2(num_classes=5, pretrained=False)
state_dict = torch.load("./gradio/models/best_doc_classifier.pth", map_location=device)
cls_model.load_state_dict(state_dict)
cls_model.to(device)
cls_model.eval()

class_labels = ["attestat", "diplom", "passport", "prilozenia", "snils"]

# YOLO-модели
yolo_models = {
    "attestat": YOLO("./gradio/models/attestat.pt"),
    "diplom": YOLO("./gradio/models/diplom.pt"),
    "passport": YOLO("./gradio/models/passport.pt"),
    "snils": YOLO("./gradio/models/snils_model.pt"),
    "prilozenia": YOLO("./gradio/models/attestat.pt")
}

# EasyOCR (твоя кастомная модель)
MODEL_DIR_PATH = "/mnt/mishutqa/PycharmProjects/sirius/gradio/custom_EasyOCR"
reader = easyocr.Reader(
    ['ru'],
    model_storage_directory=f"{MODEL_DIR_PATH}/model",
    user_network_directory=f"{MODEL_DIR_PATH}/user_network",
    recog_network='custom_example',
    # detector=False,
    gpu=True,
    download_enabled=False
)

# Трансформации для классификатора
cls_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ========== 2. Вспомогательные функции ==========

def classify_image(image):
    img_tensor = cls_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = cls_model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        return class_labels[pred_idx]


def crop_by_obb(image, obb, doc_label="document", zone_name="zone", output_dir_base="output"):
    """
    Вырезает регион по OBB, поворачивает, если он вертикальный, сохраняет и возвращает изображение.
    """
    pts = np.array(obb, dtype=np.float32).reshape(4, 2)

    # Упорядочиваем точки: tl, tr, br, bl
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl

    (tl, tr, br, bl) = rect
    width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

    if width <= 0 or height <= 0:
        return None

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))

    # === ПОВОРОТ ДЛЯ ВЕРТИКАЛЬНЫХ ФРАГМЕНТОВ ===
    if warped is not None and warped.size > 0:
        h, w = warped.shape[:2]
        if h > w:  # высота > ширины → вертикальный текст
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # === СОХРАНЕНИЕ В ФАЙЛ ===
    output_subdir = os.path.join(output_dir_base, doc_label)
    os.makedirs(output_subdir, exist_ok=True)

    existing_files = [f for f in os.listdir(output_subdir) if f.startswith(zone_name)]
    next_idx = len(existing_files)
    save_path = os.path.join(output_subdir, f"{zone_name}_{next_idx:03d}.jpg")
    cv2.imwrite(save_path, warped)

    return warped


def yolo_ocr_to_json(cls_label, yolo_results, image):
    zones_json = []

    for det in yolo_results[0].obb:
        obb = det.xyxyxyxy.cpu().numpy().flatten().tolist()
        zone_id = int(det.cls)
        zone_name = yolo_models[cls_label].names[zone_id]

        cropped = crop_by_obb(image, obb, cls_label)
        if cropped is None or cropped.size == 0:
            continue

        # Для stamp/gerb — не нужно OCR
        if zone_name in ('stamp', 'gerb'):
            zones_json.append({zone_name: True})
            continue

        ocr_result = reader.readtext(cropped)

        if not ocr_result:
            continue

        # Средняя уверенность по всем строкам в зоне
        pred_text = " ".join([text for (_, text, _) in ocr_result]).strip()

        # print("\n".join([text for (_, text, _) in ocr_result]).strip(), angle, mean_conf)
        # Если ни в одной ориентации текст не найден — оставляем пустую строку

        zones_json.append({zone_name: pred_text})

    return {
        "document": {
            "doc_name": cls_label,
            "zones": zones_json
        }
    }


# ========== 3. Основной пайплайн ==========
def process_image(image):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 1. Классификация
    cls_label = classify_image(image)

    rotated_image = preprocess_document_image(image_bgr)

    # 2. YOLO OBB
    yolo_model = yolo_models.get(cls_label)
    results = yolo_model(rotated_image)
    print(results[0].obb.conf)
    print(torch.mean(results[0].obb.conf))


    # 3. OCR + JSON
    ocr_results = yolo_ocr_to_json(cls_label, results, rotated_image)

    return ocr_results

# ========== 4. Gradio интерфейс ==========


iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Загрузите изображение"),
    outputs=gr.JSON(label="Результат"),
    title="🧠 OCR-сервис с классификацией и OBB",
    description="MobileNet → YOLOv8 OBB → EasyOCR (кастомная модель)"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7861)