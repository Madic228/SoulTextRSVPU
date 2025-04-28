import json
import pandas as pd
import re
from tqdm import tqdm

def clean_text(text):
    """
    Очищает текст от специальных символов и лишних пробелов

    Args:
        text: исходный текст

    Returns:
        очищенный текст
    """
    # Приводим к нижнему регистру
    text = text.lower()

    # Удаляем специальные символы и цифры
    text = re.sub(r'[^а-яА-Яa-zA-Z\s]', ' ', text)

    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text)

    # Удаляем пробелы в начале и конце
    text = text.strip()

    return text

def normalize_dataset(input_file, output_file, min_length=100, max_length=5000):
    """
    Нормализует датасет, удаляя слишком короткие и длинные тексты,
    а также очищая тексты от специальных символов

    Args:
        input_file: путь к входному JSON файлу
        output_file: путь к выходному JSON файлу
        min_length: минимальная длина текста
        max_length: максимальная длина текста
    """
    # Читаем исходный датасет
    print("Чтение датасета...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Преобразуем в DataFrame
    df = pd.DataFrame(data, columns=['text', 'label'])

    print(f"Исходный размер датасета: {len(df)} записей")

    # Очищаем тексты
    print("Очистка текстов...")
    df['text'] = df['text'].apply(clean_text)

    # Добавляем колонку с длиной текста
    df['text_length'] = df['text'].str.len()

    # Удаляем слишком короткие и длинные тексты
    df = df[(df['text_length'] >= min_length) & (df['text_length'] <= max_length)]

    print(f"Размер датасета после фильтрации: {len(df)} записей")

    # Анализ распределения классов
    class_distribution = df['label'].value_counts()
    total = len(df)

    print("\nРаспределение классов после нормализации:")
    print("-" * 50)
    for label, count in class_distribution.items():
        percentage = round((count / total * 100), 2)
        print(f"Класс '{label}': {count} текстов ({percentage}%)")

    # Сохраняем нормализованный датасет
    print("\nСохранение нормализованного датасета...")
    normalized_data = df[['text', 'label']].values.tolist()
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=2)

    print(f"Нормализованный датасет сохранен в {output_file}")

if __name__ == "__main__":
    input_file = "RU_dataset.json"
    output_file = "RU_dataset_normalized.json"
    normalize_dataset(input_file, output_file)