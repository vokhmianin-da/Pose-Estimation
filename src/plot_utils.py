# -*- coding: utf-8 -*-
"""
Функции для построения и сохранения графиков метрик.
"""
import matplotlib.pyplot as plt
import numpy as np

def save_metrics_plot(timestamps, cos_scores, weighted_dists, output_path='data/metrics_plot.png'):
    """
    Сохраняет график изменения метрик по времени.
    """
    plt.figure(figsize=(12, 5))

    # График косинусного сходства
    plt.subplot(1, 2, 1)
    plt.plot(timestamps, cos_scores, marker='o', linestyle='-', color='green')
    plt.xlabel('Время (с)')
    plt.ylabel('Косинусное сходство')
    plt.title('Косинусное сходство по кадрам')
    plt.grid(True)
    plt.ticklabel_format(style='plain', axis='y')  # отключаем научную нотацию

    # График взвешенного расстояния
    plt.subplot(1, 2, 2)
    plt.plot(timestamps, weighted_dists, marker='o', linestyle='-', color='red')
    plt.xlabel('Время (с)')
    plt.ylabel('Взвешенное расстояние')
    plt.title('Взвешенное расстояние по кадрам')
    plt.grid(True)
    plt.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"График сохранён в {output_path}")