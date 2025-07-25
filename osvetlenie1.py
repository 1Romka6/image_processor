import os
import argparse
import cv2
import numpy as np
import json
from tqdm import tqdm
import logging
import time
from datetime import datetime

# =============================================================================
# Конфигурация обработки по умолчанию
# =============================================================================
DEFAULT_CONFIG = {
    "INPUT_FOLDER": "134_039_109",
    "OUTPUT_FOLDER": "ready",
    "EXPOSURE": 100 / 97,  # 90% exposure
    "CONTRAST": 10,  # contrast adjustment
    "GAMMA": 3,  # gamma correction
    "CURVE_POINTS": [  # custom curve points
        [0.0, 0.0],
        [0.347985, 0.233766],
        [0.716912, 0.5],
        [0.898897, 0.915584],
        [0.977941, 0.993506],
        [1.0, 1.0]
    ],
    "DESATURATE": True,  # HLS saturation=-127
    "JPEG_QUALITY": 100,  # максимальное качество
    "PROCESS_SUBFOLDERS": True,  # обрабатывать вложенные папки
    "LOG_LEVEL": "INFO"  # Уровень логирования: DEBUG, INFO, WARNING, ERROR
}


# =============================================================================
# Функции обработки изображений
# =============================================================================
def apply_curve(image, points):
    """Применение кастомной тоновой кривой"""
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    lut = np.interp(np.linspace(0, 1, 256), x, y).clip(0, 1)
    lut = (lut * 255).astype(np.uint8)
    return cv2.LUT(image, lut)


def process_image(image, config):
    """Применяет все этапы обработки к изображению"""
    # 1. Коррекция экспозиции
    processed = cv2.convertScaleAbs(image, alpha=config["EXPOSURE"], beta=0)

    # 2. Регулировка контраста
    processed = cv2.convertScaleAbs(processed, alpha=1 + config["CONTRAST"] / 100, beta=0)

    # 3. Гамма-коррекция
    if config["GAMMA"] != 1.0:
        inv_gamma = 1.0 / config["GAMMA"]
        lut = np.array([(i / 255.0) ** inv_gamma * 255
                        for i in range(256)]).astype("uint8")
        processed = cv2.LUT(processed, lut)

    # 4. Применение тоновой кривой
    processed = apply_curve(processed, config["CURVE_POINTS"])

    # 5. Обесцвечивание
    if config["DESATURATE"]:
        hls = cv2.cvtColor(processed, cv2.COLOR_BGR2HLS)
        hls[:, :, 2] = 0  # saturation channel
        processed = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)

    # 6. Конвертация в grayscale
    return cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)


# =============================================================================
# Управляющие функции
# =============================================================================
def process_directory(input_path, output_path, config, logger):
    """Обрабатывает все изображения в директории"""
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    processed_count = 0
    error_count = 0

    # Собираем список всех файлов для обработки
    all_files = []
    for root, dirs, files in os.walk(input_path):
        rel_path = os.path.relpath(root, input_path)
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_ext:
                all_files.append((root, rel_path, file))

    total_files = len(all_files)
    logger.info(f"Найдено {total_files} файлов для обработки в {input_path}")

    if total_files == 0:
        logger.warning("Нет файлов для обработки!")
        return processed_count, error_count

    # Прогресс-бар
    with tqdm(total=total_files, desc="Обработка изображений", unit="file") as pbar:
        for root, rel_path, file in all_files:
            input_file = os.path.join(root, file)
            output_dir = os.path.join(output_path, rel_path)
            output_file = os.path.join(output_dir, file)

            try:
                # Логируем начало обработки
                logger.debug(f"Начало обработки: {input_file}")

                # Создаем директории если нужно
                os.makedirs(output_dir, exist_ok=True)

                # Обработка изображения
                image = cv2.imread(input_file)
                if image is None:
                    error_msg = f"Ошибка чтения: {input_file}"
                    logger.error(error_msg)
                    error_count += 1
                    pbar.update(1)
                    continue

                processed = process_image(image, config)
                cv2.imwrite(output_file, processed,
                            [int(cv2.IMWRITE_JPEG_QUALITY), config["JPEG_QUALITY"]])

                # Логируем успешное завершение
                logger.info(f"Обработан: {input_file} -> {output_file}")
                processed_count += 1

            except Exception as e:
                error_msg = f"Ошибка при обработке {input_file}: {str(e)}"
                logger.exception(error_msg)
                error_count += 1

            pbar.update(1)

    return processed_count, error_count


def setup_logger(config):
    """Настройка системы логирования"""
    log_level = getattr(logging, config.get("LOG_LEVEL", "INFO").upper(), logging.INFO)

    logger = logging.getLogger("image_processor")
    logger.setLevel(log_level)

    # Формат сообщений
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Консольный вывод
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Файловый вывод
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"processing_{timestamp}.log")

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_config(config_path):
    """Загружает конфигурацию из JSON файла"""
    if not os.path.exists(config_path):
        print(f"Конфигурационный файл {config_path} не найден. Используются настройки по умолчанию.")
        return DEFAULT_CONFIG

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Объединяем с дефолтными значениями
        return {**DEFAULT_CONFIG, **config}
    except Exception as e:
        print(f"Ошибка загрузки конфигурации: {str(e)}. Используются настройки по умолчанию.")
        return DEFAULT_CONFIG


def main():
    """Основная функция выполнения скрипта"""
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Обработка сканов')
    parser.add_argument('-i', '--input', default='input', help='Входная директория с изображениями')
    parser.add_argument('-o', '--output', default='output', help='Выходная директория')
    parser.add_argument('-c', '--config', default='config.json', help='Путь к файлу конфигурации')
    args = parser.parse_args()

    # Загрузка конфигурации
    config = load_config(args.config)

    # Настройка логирования
    logger = setup_logger(config)

    logger.info("=" * 60)
    logger.info(f"Запуск обработки изображений")
    logger.info(f"Входная директория: {args.input}")
    logger.info(f"Выходная директория: {args.output}")
    logger.info("Параметры обработки:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    # Проверка существования входной директории
    if not os.path.exists(args.input):
        logger.error(f"Входная директория не существует: {args.input}")
        return

    # Создание выходной директории
    os.makedirs(args.output, exist_ok=True)

    # Обработка изображений
    processed_count, error_count = process_directory(
        args.input,
        args.output,
        config,
        logger
    )

    # Расчет времени выполнения
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

    logger.info("\n" + "=" * 60)
    logger.info(f"ОБРАБОТКА ЗАВЕРШЕНА")
    logger.info(f"Всего файлов: {processed_count + error_count}")
    logger.info(f"Успешно обработано: {processed_count}")
    logger.info(f"С ошибками: {error_count}")
    logger.info(f"Общее время выполнения: {time_str}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()