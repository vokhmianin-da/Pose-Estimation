import argparse
import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pose_utils import draw_two_skeletons
from src.comparison import get_transform, apply_transform, cosine_distance, weighted_distance
from src.video_processor import process_video
from myproject_models.keypoint_rcnn import KeypointRCNNModel
import scripts.config as cfg

def get_grade(mean_cos, mean_wdist):
    """
    Возвращает текстовую оценку на основе средних метрик.
    Пороги можно настроить под конкретное упражнение.
    """
    if mean_cos > 0.98 and mean_wdist < 20:
        return "Cool"
    elif mean_cos > 0.95 and mean_wdist < 30:
        return "Good"
    elif mean_cos > 0.9 and mean_wdist < 50:
        return "Middle"
    else:
        return "Fail"

def main():
    parser = argparse.ArgumentParser(description='Сравнение двух видео с упражнениями')
    parser.add_argument('--ref', required=True, help='Путь к референсному видео (коуч)')
    parser.add_argument('--user', required=True, help='Путь к видео пользователя')
    parser.add_argument('--output', default='result.mp4', help='Путь для сохранения выходного видео')
    parser.add_argument('--step', type=int, default=cfg.FRAME_STEP, help='Шаг обработки кадров')
    parser.add_argument('--conf', type=float, default=cfg.CONF_THRESHOLD, help='Порог уверенности детекции')
    parser.add_argument('--both', action='store_true', help='Накладывать оба скелета (коуч и пользователь)')
    parser.add_argument('--progress', action='store_true', help='Показывать прогресс-бары')
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
    ref_pts, ref_conf, ref_time, ref_frames = process_video(
        args.ref, model.model, model.device,
        frame_step=args.step, conf_threshold=args.conf,
        progress=progress, desc="Референс"
    )
    print(f"Получено кадров с людьми: {len(ref_pts)}")

    print(f"Обработка видео пользователя: {args.user}")
    user_pts, user_conf, user_time, user_frames = process_video(
        args.user, model.model, model.device,
        frame_step=args.step, conf_threshold=args.conf,
        progress=progress, desc="Пользователь"
    )
    print(f"Получено кадров с людьми: {len(user_pts)}")

    if len(ref_pts) == 0 or len(user_pts) == 0:
        print("Ошибка: нет кадров с обнаруженными людьми в одном из видео.")
        return

    # Сопоставляем по минимальному числу кадров
    n_frames = min(len(ref_pts), len(user_pts))
    cos_scores = []
    weighted_dists = []
    transforms = []  # матрицы преобразования для каждого кадра

    print("Вычисление метрик...")
    iterator = range(n_frames)
    if progress is not None:
        iterator = progress(iterator, desc="Метрики", unit="кадров")

    for i in iterator:
        ref = ref_pts[i]
        user = user_pts[i]
        user_conf_i = user_conf[i]

        # Получаем матрицу преобразования из user в ref
        A = get_transform(user, ref)
        transforms.append(A)

        # Применяем преобразование к точкам пользователя
        user_aligned = apply_transform(user, A)

        # Метрики
        cos_val = cosine_distance(ref.flatten(), user_aligned.flatten())
        w_dist = weighted_distance(user_aligned.flatten(), ref.flatten(), user_conf_i)

        cos_scores.append(cos_val)
        weighted_dists.append(w_dist)

    mean_cos = np.mean(cos_scores)
    mean_wdist = np.mean(weighted_dists)
    grade = get_grade(mean_cos, mean_wdist)

    print(f"\nСреднее косинусное сходство: {mean_cos:.4f}")
    print(f"Среднее взвешенное расстояние: {mean_wdist:.2f}")
    print(f"Оценка: {grade}")

    # --- Создание выходного видео ---
    print("Создание выходного видео...")
    cap_user = cv2.VideoCapture(args.user)
    fps_user = cap_user.get(cv2.CAP_PROP_FPS)
    width = int(cap_user.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_user.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps_user, (width, height))

    # Словарь для быстрого доступа к данным по номеру кадра
    frame_to_data = {}
    for i, frame_idx in enumerate(user_frames[:n_frames]):
        frame_to_data[frame_idx] = {
            'ref_pts': ref_pts[i],
            'user_pts': user_pts[i],
            'user_conf': user_conf[i],
            'transform': transforms[i],
            'cos': cos_scores[i],
            'wdist': weighted_dists[i]
        }

    total_frames_user = int(cap_user.get(cv2.CAP_PROP_FRAME_COUNT))
    iterator_frames = range(total_frames_user)
    if progress is not None:
        iterator_frames = progress(iterator_frames, desc="Запись видео", unit="кадров")

    prev_data = None
    for frame_count in iterator_frames:
        ret, frame = cap_user.read()
        if not ret:
            break

        # Если для этого кадра есть данные, накладываем скелет(ы)
        if frame_count in frame_to_data:
            data = frame_to_data[frame_count]
            prev_data = data
        else:
            data = prev_data
        if data is None:
            # Если для кадра нет данных, просто копируем исходный кадр (без скелетов)
            out.write(frame)
            
        # Преобразуем точки коуча в систему координат пользователя
        # У нас есть матрица A, переводящая user -> ref. Нужно обратное преобразование ref -> user.
        # Можно получить inv(A) и применить к ref_pts.
        # Проще: у нас уже есть user_aligned, но это user -> ref. А нам нужно ref наложить на user.
        # Найдём обратное преобразование: из ref в user.
        # Т.к. A переводит user (pad) в ref (pad), то обратное преобразование B = inv(A) (если A обратима).
        # Используем псевдообратную.
        A = data['transform']
        # Псевдообратная для однородных координат
        try:
            A_inv = np.linalg.pinv(A)  # 3x3
        except:
            A_inv = np.eye(3)
        ref_on_user = apply_transform(data['ref_pts'], A_inv)

        # Рисуем скелет коуча (зелёный)
        frame_with_skels = draw_two_skeletons(
            frame,
            ref_on_user, np.ones(17),  # для коуча все точки считаем видимыми
            data['user_pts'] if args.both else np.zeros((17,2)),
            data['user_conf'] if args.both else np.zeros(17),
            color1=(0,255,0), color2=(0,0,255),
            threshold=cfg.KEYPOINT_THRESHOLD
        ) if args.both else draw_two_skeletons(
            frame,
            ref_on_user, np.ones(17),
            np.zeros((17,2)), np.zeros(17),
            color1=(0,255,0), color2=(0,0,255),
            threshold=cfg.KEYPOINT_THRESHOLD
        )
        # Добавляем текст с метриками
        cv2.putText(frame_with_skels, f'Cos: {data["cos"]:.3f}', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame_with_skels, f'WDist: {data["wdist"]:.2f}', (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        out.write(frame_with_skels)
        """else:
            # Если для кадра нет данных, просто копируем исходный кадр (без скелетов)
            out.write(frame)"""

    cap_user.release()

    # Добавляем итоговый кадр (чёрный с текстом)
    summary_frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(summary_frame, f"Mean Cosine Similarity: {mean_cos:.4f}", (50, height//2 - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(summary_frame, f"Mean Weighted Distance: {mean_wdist:.2f}", (50, height//2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(summary_frame, f"Grade: {grade}", (50, height//2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    # Дублируем кадр в течение 1 секунды (округляем до целого числа кадров)
    for _ in range(int(fps_user)):
        out.write(summary_frame)

    out.release()
    print(f"Выходное видео сохранено: {args.output}")

if __name__ == '__main__':
    main()