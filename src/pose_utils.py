# -*- coding: utf-8 -*-
"""
Утилиты для работы с ключевыми точками и отрисовки скелета.
"""
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Список названий ключевых точек COCO (индексы 0..16)
KEYPOINT_NAMES = [
    'нос', 'левый глаз', 'правый глаз', 'левое ухо', 'правое ухо',
    'левое плечо', 'правое плечо', 'левый локоть', 'правый локоть',
    'левое запястье', 'правое запястье', 'левое бедро', 'правое бедро',
    'левое колено', 'правое колено', 'левая лодыжка', 'правая лодыжка'
]

# Соединения для скелета (индексы из KEYPOINT_NAMES)
LIMBS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # голова
    (5, 7), (7, 9), (6, 8), (8, 10),         # руки
    (11, 13), (13, 15), (12, 14), (14, 16),  # ноги
    (5, 6), (11, 12), (5, 11), (6, 12)       # торс
]

def load_model(device='cuda'):
    """
    Загружает предобученную модель Keypoint R-CNN и переводит в режим eval.
    """
    import torchvision
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model.to(device)

def get_keypoints_from_pil(img_pil, model, device, conf_threshold=0.9):
    """
    Принимает PIL image, модель и порог.
    Возвращает (pts, scores, conf) для лучшего обнаруженного человека.
    pts: (17,2) координаты
    scores: (17,) достоверности ключевых точек
    conf: уверенность детекции человека
    """
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
    output = outputs[0]
    boxes_scores = output['scores'].cpu().numpy()
    keypoints_all = output['keypoints'].cpu().numpy()          # [N, 17, 3]
    keypoints_scores = output['keypoints_scores'].cpu().numpy()  # [N, 17]

    # Отбираем детекции с уверенностью выше порога
    valid_idx = np.where(boxes_scores > conf_threshold)[0]
    if len(valid_idx) == 0:
        return None, None, None
    # Берём самую уверенную
    best_idx = valid_idx[np.argmax(boxes_scores[valid_idx])]
    pts = keypoints_all[best_idx, :, :2]      # (17,2)
    scores = keypoints_scores[best_idx]       # (17,)
    conf = boxes_scores[best_idx]
    return pts, scores, conf

def draw_skeleton(img, keypoints, scores, threshold=0.5, color=(0,255,0)):
    """
    Рисует скелет на изображении (BGR).
    img: изображение в формате BGR (numpy array)
    keypoints: (17,2)
    scores: (17,)
    threshold: порог для отображения точки/соединения
    color: цвет в BGR
    Возвращает копию изображения.
    """
    img_copy = img.copy()
    # Рисуем соединения
    for (idx1, idx2) in LIMBS:
        if scores[idx1] > threshold and scores[idx2] > threshold:
            x1, y1 = int(keypoints[idx1, 0]), int(keypoints[idx1, 1])
            x2, y2 = int(keypoints[idx2, 0]), int(keypoints[idx2, 1])
            cv2.line(img_copy, (x1, y1), (x2, y2), color, 2)
    # Рисуем точки
    for i, pt in enumerate(keypoints):
        if scores[i] > threshold:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(img_copy, (x, y), 4, color, -1)
    return img_copy

def draw_two_skeletons(img, keypoints1, scores1, keypoints2, scores2,
                       color1=(0,255,0), color2=(0,0,255), threshold=0.5):
    """
    Рисует два скелета на одном изображении.
    keypoints1/2: (17,2)
    scores1/2: (17,)
    color1/2: цвета в BGR
    """
    img_copy = img.copy()
    # Первый скелет
    for (idx1, idx2) in LIMBS:
        if scores1[idx1] > threshold and scores1[idx2] > threshold:
            x1, y1 = int(keypoints1[idx1,0]), int(keypoints1[idx1,1])
            x2, y2 = int(keypoints1[idx2,0]), int(keypoints1[idx2,1])
            cv2.line(img_copy, (x1,y1), (x2,y2), color1, 2)
    for i, pt in enumerate(keypoints1):
        if scores1[i] > threshold:
            x,y = int(pt[0]), int(pt[1])
            cv2.circle(img_copy, (x,y), 4, color1, -1)
    # Второй скелет
    for (idx1, idx2) in LIMBS:
        if scores2[idx1] > threshold and scores2[idx2] > threshold:
            x1, y1 = int(keypoints2[idx1,0]), int(keypoints2[idx1,1])
            x2, y2 = int(keypoints2[idx2,0]), int(keypoints2[idx2,1])
            cv2.line(img_copy, (x1,y1), (x2,y2), color2, 2)
    for i, pt in enumerate(keypoints2):
        if scores2[i] > threshold:
            x,y = int(pt[0]), int(pt[1])
            cv2.circle(img_copy, (x,y), 4, color2, -1)
    return img_copy