import argparse
import sys
import subprocess
import os

def trim_video_ffmpeg(input_file: str, output_file: str, start_time: float, end_time: float):
    # Проверяем наличие входного файла
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Входной файл '{input_file}' не найден.")
    
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-ss', str(start_time),
        '-to', str(end_time),
        '-c', 'copy',
        '-y',
        output_file
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg ошибка: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg не найден. Убедитесь, что ffmpeg установлен и доступен в PATH.")

def main():
    parser = argparse.ArgumentParser(
        description="Обрезает видео по заданным временным меткам (в секундах)."
    )
    parser.add_argument("input", help="Путь к исходному видеофайлу")
    parser.add_argument("output", help="Путь для сохранения обрезанного видео")
    parser.add_argument("start", type=float, help="Начало фрагмента в секундах")
    parser.add_argument("end", type=float, help="Конец фрагмента в секундах")

    args = parser.parse_args()

    if args.start >= args.end:
        print("Ошибка: начало должно быть строго меньше конца.", file=sys.stderr)
        sys.exit(1)

    try:
        trim_video_ffmpeg(args.input, args.output, args.start, args.end)
        print(f"Видео успешно обрезано и сохранено в {args.output}")
    except FileNotFoundError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Ошибка при обработке видео: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Неожиданная ошибка: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()