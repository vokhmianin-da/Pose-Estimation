# -*- coding: utf-8 -*-
"""
Основной скрипт для сравнения двух видео.
Поддерживает отображение скелета коуча (флаг --skelet), русский текст, сохранение графиков.
"""
import argparse
import cv2
import numpy as np
import sys
import os
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pose_utils import draw_two_skeletons
from src.comparison import get_transform, apply_transform, cosine_distance, weighted_distance
from src.video_processor import process_video
from src.plot_utils import save_metrics_plot
from myproject_models.keypoint_rcnn import KeypointRCNNModel
import scripts.config as cfg

def interpolate_points(pts_list, timestamps, new_timestamps):
    """
    Линейная интерполяция ключевых точек.
    pts_list: список (N, 17, 2)
    timestamps: (N,) исходные времена
    new_timestamps: (M,) новые времена
    Возвращает список (M, 17, 2)
    """
    if len(pts_list) == 0:
        return []
    pts_array = np.array(pts_list)  # (N, 17, 2)
    N, J, D = pts_array.shape
    flat = pts_array.reshape(N, -1)  # (N, 34)
    interpolated = []
    for t in new_timestamps:
        if t <= timestamps[0]:
            interpolated.append(pts_list[0])
        elif t >= timestamps[-1]:
            interpolated.append(pts_list[-1])
        else:
            idx = np.searchsorted(timestamps, t)
            t0, t1 = timestamps[idx-1], timestamps[idx]
            pt0 = flat[idx-1]
            pt1 = flat[idx]
            alpha = (t - t0) / (t1 - t0)
            pt_interp = pt0 + alpha * (pt1 - pt0)
            interpolated.append(pt_interp.reshape(J, D))
    return interpolated

def put_text_ru(img, text, position, font_size=30, color=(255,255,255), font_path=None):
    """
    Рисует текст на изображении с поддержкой кириллицы с помощью Pillow.
    img: изображение в формате BGR (numpy array)
    text: строка для отображения (может содержать русские буквы)
    position: (x, y) верхнего левого угла
    font_size: размер шрифта
    color: цвет текста в BGR
    font_path: путь к ttf-шрифту (если None, используется шрифт по умолчанию)
    Возвращает изображение BGR.
    """
    # Конвертируем BGR в RGB для PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    # Пытаемся загрузить шрифт
    try:
        if font_path is None:
            # Для Windows часто доступен Arial
            font = ImageFont.truetype("arial.ttf", font_size)
        else:
            font = ImageFont.truetype(font_path, font_size)
    except:
        # Если шрифт не найден, используем стандартный (без поддержки кириллицы)
        font = ImageFont.load_default()
        print("Предупреждение: не удалось загрузить шрифт, русский текст может отображаться некорректно.")
    # Рисуем текст
    draw.text(position, text, font=font, fill=color[::-1])  # color в RGB
    # Обратно в BGR
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def get_grade(mean_cos, mean_wdist, russian=False):
    """
    Возвращает текстовую оценку на основе средних метрик.
    """
    if mean_cos > 0.98 and mean_wdist < 20:
        return "Отлично" if russian else "Excellent"
    elif mean_cos > 0.95 and mean_wdist < 30:
        return "Хорошо" if russian else "Good"
    elif mean_cos > 0.9 and mean_wdist < 50:
        return "Средне" if russian else "Average"
    else:
        return "Плохо" if russian else "Poor"

def main():
    parser = argparse.ArgumentParser(description='Сравнение двух видео с упражнениями')
    parser.add_argument('--ref', required=True, help='Путь к референсному видео (коуч)')
    parser.add_argument('--user', required=True, help='Путь к видео пользователя')
    parser.add_argument('--output', default='result.mp4', help='Путь для сохранения выходного видео')
    parser.add_argument('--step', type=int, default=cfg.FRAME_STEP, help='Шаг обработки кадров (каждый N-й кадр)')
    parser.add_argument('--conf', type=float, default=cfg.CONF_THRESHOLD, help='Порог уверенности детекции (0..1)')
    parser.add_argument('--skelet', action='store_true', help='Накладывать скелет коуча (зелёный) на видео пользователя. Если не указан, выводятся только метрики.')
    parser.add_argument('--progress', action='store_true', help='Показывать прогресс-бары')
    parser.add_argument('--ru', action='store_true', help='Использовать русский язык для итогового текста')
    parser.add_argument('--plot', help='Сохранить графики метрик в указанный файл')
    args = parser.parse_args()

    # Настройка прогресса
    if args.progress:
        try:
            from tqdm import tqdm
            progress = tqdm
        except ImportError:
            print("Библиотека tqdm не установлена. Установите: pip install tqdm")
            progress = None
    else:
        progress = None

    # Загружаем модель
    print("Загрузка модели...")
    model = KeypointRCNNModel(device=cfg.DEVICE)

    # Обрабатываем видео
    print(f"Обработка референсного видео: {args.ref}")
    ref_pts, ref_conf, ref_time, ref_frames, ref_fps = process_video(
        args.ref, model.model, model.device,
        frame_step=args.step, conf_threshold=args.conf,
        progress=progress, desc="Референс"
    )
    print(f"Получено кадров с людьми: {len(ref_pts)}")

    print(f"Обработка видео пользователя: {args.user}")
    user_pts, user_conf, user_time, user_frames, user_fps = process_video(
        args.user, model.model, model.device,
        frame_step=args.step, conf_threshold=args.conf,
        progress=progress, desc="Пользователь"
    )
    print(f"Получено кадров с людьми: {len(user_pts)}")

    if len(ref_pts) == 0 or len(user_pts) == 0:
        print("Ошибка: нет кадров с обнаруженными людьми в одном из видео.")
        return

    # Создаём общую временную шкалу (используем FPS пользователя)
    fps_common = user_fps
    max_time = min(ref_time[-1], user_time[-1])
    common_timestamps = np.arange(0, max_time, 1.0/fps_common)

    print("Интерполяция ключевых точек на общее время...")
    ref_pts_interp = interpolate_points(ref_pts, ref_time, common_timestamps)
    user_pts_interp = interpolate_points(user_pts, user_time, common_timestamps)

    # Интерполяция достоверностей (ближайший сосед)
    user_conf_interp = []
    for t in common_timestamps:
        idx = np.searchsorted(user_time, t)
        if idx == 0:
            user_conf_interp.append(user_conf[0])
        elif idx >= len(user_time):
            user_conf_interp.append(user_conf[-1])
        else:
            if t - user_time[idx-1] < user_time[idx] - t:
                user_conf_interp.append(user_conf[idx-1])
            else:
                user_conf_interp.append(user_conf[idx])

    n_frames = len(common_timestamps)
    cos_scores = []
    weighted_dists = []
    transforms_ref_to_user = []  # матрица для преобразования референса в координаты пользователя

    print("Вычисление метрик и преобразований...")
    iterator = range(n_frames)
    if progress is not None:
        iterator = progress(iterator, desc="Метрики")

    for i in iterator:
        ref = ref_pts_interp[i]
        user = user_pts_interp[i]
        user_conf_i = user_conf_interp[i]

        # Преобразование user -> ref (для метрик)
        A = get_transform(user, ref)
        user_aligned = apply_transform(user, A)
        cos_val = cosine_distance(ref.flatten(), user_aligned.flatten())
        w_dist = weighted_distance(user_aligned.flatten(), ref.flatten(), user_conf_i)

        # Преобразование ref -> user (для наложения скелета коуча)
        B = get_transform(ref, user)
        transforms_ref_to_user.append(B)

        cos_scores.append(cos_val)
        weighted_dists.append(w_dist)

    mean_cos = np.mean(cos_scores)
    mean_wdist = np.mean(weighted_dists)
    grade = get_grade(mean_cos, mean_wdist, russian=args.ru)

    print(f"\nСреднее косинусное сходство: {mean_cos:.4f}")
    print(f"Среднее взвешенное расстояние: {mean_wdist:.2f}")
    print(f"Оценка: {grade}")

    # Сохранение графиков, если запрошено
    if args.plot:
        save_metrics_plot(common_timestamps, cos_scores, weighted_dists, args.plot)

    # --- Создание выходного видео ---
    print("Создание выходного видео...")
    cap_user = cv2.VideoCapture(args.user)
    width = int(cap_user.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_user.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_user_frames = int(cap_user.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, user_fps, (width, height))

    # Сопоставляем каждый кадр пользователя с ближайшим общим временным индексом
    frame_to_data = {}
    for frame_idx in range(total_user_frames):
        t = frame_idx / user_fps
        if t > max_time:
            continue
        common_idx = np.argmin(np.abs(common_timestamps - t))
        frame_to_data[frame_idx] = {
            'ref_pts': ref_pts_interp[common_idx],
            'B': transforms_ref_to_user[common_idx],  # матрица ref->user
            'cos': cos_scores[common_idx],
            'wdist': weighted_dists[common_idx]
        }

    iterator_frames = range(total_user_frames)
    if progress is not None:
        iterator_frames = progress(iterator_frames, desc="Запись видео")

    for frame_count in iterator_frames:
        ret, frame = cap_user.read()
        if not ret:
            break

        if frame_count in frame_to_data:
            data = frame_to_data[frame_count]

            # Если запрошен скелет, преобразуем точки референса и рисуем
            if args.skelet:
                ref_on_user = apply_transform(data['ref_pts'], data['B'])
                frame_with_skels = draw_two_skeletons(
                    frame,
                    ref_on_user, np.ones(17),  # для референса все точки считаем видимыми
                    np.zeros((17,2)), np.zeros(17),  # скелет пользователя не рисуем
                    color1=(0,255,0), color2=(0,0,255),
                    threshold=cfg.KEYPOINT_THRESHOLD
                )
            else:
                frame_with_skels = frame.copy()

            # Добавляем текст с метриками (всегда)
            cv2.putText(frame_with_skels, f'Cos: {data["cos"]:.3f}', (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame_with_skels, f'WDist: {data["wdist"]:.2f}', (10,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            out.write(frame_with_skels)
        else:
            # Кадры без данных просто копируем (без метрик)
            out.write(frame)

    cap_user.release()

    # Итоговый кадр с оценкой
    summary_frame = np.zeros((height, width, 3), dtype=np.uint8)
    if args.ru:
        summary_frame = put_text_ru(summary_frame, f"Ср. косинус: {mean_cos:.4f}", (50, height//2 - 60), font_size=30)
        summary_frame = put_text_ru(summary_frame, f"Ср. расст.: {mean_wdist:.2f}", (50, height//2 - 20), font_size=30)
        summary_frame = put_text_ru(summary_frame, f"Оценка: {grade}", (50, height//2 + 20), font_size=30)
    else:
        cv2.putText(summary_frame, f"Mean Cosine: {mean_cos:.4f}", (50, height//2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(summary_frame, f"Mean WDist: {mean_wdist:.2f}", (50, height//2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(summary_frame, f"Grade: {grade}", (50, height//2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    for _ in range(int(user_fps)):
        out.write(summary_frame)

    out.release()
    print(f"Выходное видео сохранено: {args.output}")

if __name__ == '__main__':
    main()