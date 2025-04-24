import json
import pandas as pd
import numpy as np

def analyze_text_lengths(df, n=15):
    """
    Анализирует самые длинные и самые короткие тексты в датасете
    
    Args:
        df: DataFrame с текстами
        n: количество текстов для вывода
    """
    # Создаем копию DataFrame с длиной текстов
    df_with_length = df.copy()
    df_with_length['text_length'] = df_with_length['text'].str.len()
    
    # Получаем самые длинные тексты
    longest_texts = df_with_length.nlargest(n, 'text_length')
    print(f"\n{n} самых длинных текстов:")
    print("-" * 50)
    for idx, row in longest_texts.iterrows():
        print(f"\nДлина: {row['text_length']} символов")
        print(f"Класс: {row['label']}")
        print(f"Текст: {row['text'][:200]}...")  # Показываем только первые 200 символов
    
    # Получаем самые короткие тексты
    shortest_texts = df_with_length.nsmallest(n, 'text_length')
    print(f"\n{n} самых коротких текстов:")
    print("-" * 50)
    for idx, row in shortest_texts.iterrows():
        print(f"\nДлина: {row['text_length']} символов")
        print(f"Класс: {row['label']}")
        print(f"Текст: {row['text']}")

def analyze_dataset(file_path):
    # Читаем JSON файл
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Преобразуем в DataFrame с правильными названиями колонок
    df = pd.DataFrame(data, columns=['text', 'label'])
    
    # Выводим информацию о структуре данных
    print("\nИнформация о датасете:")
    print("-" * 50)
    print(df.info())
    
    # Анализ распределения классов
    print("\nРаспределение классов:")
    print("-" * 50)
    class_distribution = df['label'].value_counts()
    total = len(df)
    
    for label, count in class_distribution.items():
        percentage = round((count / total * 100), 2)
        print(f"Класс '{label}': {count} текстов ({percentage}%)")
    
    # Дополнительная статистика
    print("\nСтатистика по длине текстов:")
    print("-" * 50)
    df['text_length'] = df['text'].str.len()
    print(f"Средняя длина текста: {round(df['text_length'].mean(), 2)} символов")
    print(f"Минимальная длина: {df['text_length'].min()} символов")
    print(f"Максимальная длина: {df['text_length'].max()} символов")
    
    # Статистика по классам
    print("\nСтатистика по классам:")
    print("-" * 50)
    class_stats = df.groupby('label')['text_length'].agg(['mean', 'min', 'max'])
    print(class_stats.round(2))
    
    # Анализ самых длинных и самых коротких текстов
    analyze_text_lengths(df)

if __name__ == "__main__":
    #file_path = "RU_dataset.json"
    file_path = "RU_dataset_normalized.json"
    analyze_dataset(file_path) 