import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Настройка GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    print("Используется GPU")
    # Динамическое выделение памяти GPU
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("GPU не найден, используется CPU")

def create_model(vocab_size, max_length):
    """Создание модели"""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 128),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def load_model_and_tokenizer():
    """Загрузка модели и токенизатора"""
    # Пути к файлам
    weights_path = os.path.join(os.path.dirname(__file__), "depression_detector_weights.h5")
    tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer.pkl")

    # Создаем модель
    model = create_model(vocab_size=10000, max_length=500)
    
    # Загружаем веса
    try:
        model.load_weights(weights_path)
        print(f"Веса модели загружены из: {weights_path}")
    except FileNotFoundError:
        print(f"Ошибка: Файл весов не найден по пути: {weights_path}")
        exit()

    # Загружаем токенизатор
    try:
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        print(f"Токенизатор загружен из: {tokenizer_path}")
    except FileNotFoundError:
        print(f"Ошибка: Файл токенизатора не найден по пути: {tokenizer_path}")
        exit()

    return model, tokenizer


def predict_text(text, model, tokenizer, max_length=500):
    """Предсказание для одного текста"""
    # Токенизация и паддинг
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

    # Предсказание
    prediction = model.predict(padded)[0][0]
    return prediction

def main():
    # Загрузка модели и токенизатора
    print("Загрузка модели и токенизатора...")
    model, tokenizer = load_model_and_tokenizer()

    while True:
        # Получение текста от пользователя
        text = input("\nВведите текст для анализа (или 'q' для выхода): ")
        if text.lower() == 'q':
            break

        # Предсказание
        prediction = predict_text(text, model, tokenizer)

        # Вывод результата
        print("\nРезультат анализа:")
        print(f"Вероятность депрессивного текста: {prediction:.2%}")
        print(f"Класс: {'Депрессивный' if prediction > 0.5 else 'Не депрессивный'}")

if __name__ == "__main__":
    main()