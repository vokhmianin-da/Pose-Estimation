import cv2
import numpy as np
from PIL import Image
from .pose_utils import get_keypoints_from_pil

def process_video(video_path, model, device, frame_step=5, conf_threshold=0.9, progress=None, desc="Обработка видео"):
    """
    Извлекает ключевые точки из каждого frame_step-го кадра видео.
    Параметры:
        progress: функция для отображения прогресса (например, из tqdm), либо None
        desc: описание для прогресс-бара
    Возвращает:
        all_pts: список массивов (17,2) для каждого успешного кадра
        all_conf: список массивов (17,) достоверностей
        timestamps: список временных меток (сек) для этих кадров
        frame_indices: список номеров кадров (глобальных), на которых были обнаружены люди
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    all_pts = []
    all_conf = []
    timestamps = []
    frame_indices = []
    frame_count = 0

    if progress is not None:
        pbar = progress(total=total_frames, desc=desc, unit="frames")
    else:
        pbar = None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_count % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            pts, confs, score = get_keypoints_from_pil(img_pil, model, device, conf_threshold)
            if pts is not None:
                all_pts.append(pts)
                all_conf.append(confs)
                timestamps.append(frame_count / fps)
                frame_indices.append(frame_count)
        frame_count += 1
        if pbar is not None:
            pbar.update(1)

    cap.release()
    if pbar is not None:
        pbar.close()
    return all_pts, all_conf, timestamps, frame_indices