from pydub import AudioSegment
import noisereduce as nr
import numpy as np
from scipy.io import wavfile
import logging
from tqdm import tqdm
import os

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Конвертация MP3 в WAV
def convert_mp3_to_wav(input_mp3, output_wav):
    logging.info(f"Конвертация {input_mp3} в WAV...")
    audio = AudioSegment.from_mp3(input_mp3)
    audio.export(output_wav, format="wav")
    logging.info("Конвертация завершена.")

# Применение шумоподавления
def reduce_noise_with_progress(input_wav, output_wav):
    logging.info("Начало шумоподавления...")
    rate, data = wavfile.read(input_wav)

    if len(data.shape) > 1:  # Если стерео
        data = data[:, 0]

    # Прогресс обработки
    chunk_size = len(data) // 100
    reduced_noise = np.zeros_like(data)

    for i in tqdm(range(0, len(data), chunk_size), desc="Обработка шума"):
        chunk = data[i:i+chunk_size]
        reduced_chunk = nr.reduce_noise(y=chunk, sr=rate, prop_decrease=0.8)
        reduced_noise[i:i+chunk_size] = reduced_chunk

    wavfile.write(output_wav, rate, reduced_noise.astype(np.int16))
    logging.info("Шумоподавление завершено.")

# Конвертация обратно в MP3
def convert_wav_to_mp3(input_wav, output_mp3):
    logging.info(f"Конвертация {input_wav} обратно в MP3...")
    audio = AudioSegment.from_wav(input_wav)
    audio.export(output_mp3, format="mp3")
    logging.info("Конвертация завершена.")

# Основная функция обработки одного файла
def process_audio(input_mp3, output_mp3):
    temp_wav = "temp.wav"
    clean_wav = "clean.wav"

    try:
        convert_mp3_to_wav(input_mp3, temp_wav)
        reduce_noise_with_progress(temp_wav, clean_wav)
        convert_wav_to_mp3(clean_wav, output_mp3)
    finally:
        # Удаление временных файлов
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        if os.path.exists(clean_wav):
            os.remove(clean_wav)

    logging.info(f"Файл {output_mp3} успешно обработан!")

# Функция для автоматической обработки всех MP3-файлов в папке
def process_all_files(input_folder, output_folder):
    # Убедитесь, что папка для сохранения результатов существует
    os.makedirs(output_folder, exist_ok=True)

    # Обработка всех файлов MP3
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"cleaned_{filename}")
            logging.info(f"Обработка {filename}...")
            process_audio(input_path, output_path)
            logging.info(f"Файл {filename} обработан и сохранён как {output_path}")

# Пример использования для папки
input_folder = "input_audio"  # Папка с исходными MP3
output_folder = "output_audio"  # Папка для сохранённых файлов

# Обработка всех файлов в папке
process_all_files(input_folder, output_folder)
