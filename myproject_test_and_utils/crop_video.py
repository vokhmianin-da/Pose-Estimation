import subprocess
import os
import time

def trim_video_ffmpeg(input_file: str, output_file: str, start_time: float, end_time: float):
    """Обрезает видео через ffmpeg."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Входной файл '{input_file}' не найден.")
    
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-ss', str(start_time),
        '-to', str(end_time),
        '-c', 'copy',
        '-y',  # перезапись
        output_file
    ]
    
    # Используем with для гарантированного закрытия потоков и освобождения ресурсов
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    ) as proc:
        try:
            # Ждем завершения и проверяем код возврата
            stdout, stderr = proc.communicate(timeout=60)  # таймаут на всякий случай
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg ошибка (код {proc.returncode}): {stderr}")
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise RuntimeError("FFmpeg процесс превысил время ожидания")
    
    # Дополнительная проверка, что выходной файл создан
    if not os.path.exists(output_file):
        raise RuntimeError("Выходной файл не был создан")
    
    print(f"✅ Видео обрезано и сохранено в {output_file}")
