import numpy as np

def pad(x):
    """Добавляет столбец единиц для аффинного преобразования."""
    return np.hstack([x, np.ones((x.shape[0], 1))])

def unpad(x):
    """Удаляет последний столбец."""
    return x[:, :-1]

def align_affine(src_pts, dst_pts):
    """
    Аффинное преобразование src -> dst.
    src_pts, dst_pts: массивы (N,2)
    Возвращает преобразованные src_pts (N,2).
    """
    X = pad(src_pts)   # (N,3)
    Y = pad(dst_pts)   # (N,3)
    A, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    A[np.abs(A) < 1e-10] = 0
    transformed = unpad(np.dot(pad(src_pts), A))
    return transformed

def get_transform(src_pts, dst_pts):
    """
    Возвращает матрицу аффинного преобразования (3x3) такую, что
    pad(src_pts) @ A ≈ pad(dst_pts).
    """
    X = pad(src_pts)
    Y = pad(dst_pts)
    A, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    A[np.abs(A) < 1e-10] = 0
    return A

def apply_transform(pts, A):
    """
    Применяет матрицу преобразования A (3x3) к точкам pts (N,2).
    """
    return unpad(np.dot(pad(pts), A))

def cosine_distance(pose1, pose2):
    """
    pose1, pose2 – одномерные векторы (например, 34 элемента).
    Возвращает косинусное сходство (ближе к 1 – лучше).
    """
    v1 = pose1.flatten()
    v2 = pose2.flatten()
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_sim = np.dot(v1, v2) / (norm1 * norm2)
    return cos_sim

def weighted_distance(pose1, pose2, conf1):
    """
    Взвешенное среднее абсолютных отклонений.
    pose1, pose2 – одномерные векторы координат (x1,y1,x2,y2,...).
    conf1 – достоверности ключевых точек (17 значений).
    Возвращает средневзвешенное расстояние.
    """
    sum_conf = np.sum(conf1)
    if sum_conf == 0:
        return float('inf')
    weighted_sum = 0.0
    for i in range(len(pose1)):
        kp_idx = i // 2
        weighted_sum += conf1[kp_idx] * abs(pose1[i] - pose2[i])
    return weighted_sum / sum_conf