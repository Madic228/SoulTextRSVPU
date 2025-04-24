import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def load_model_and_tokenizer():
    """Загрузка модели и токенизатора"""
    model = tf.keras.models.load_model("depression_detector.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
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