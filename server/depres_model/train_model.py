import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import matplotlib.pyplot as plt

class F1ScoreCallback(Callback):
    def __init__(self, validation_data):
        super(F1ScoreCallback, self).__init__()
        self.validation_data = validation_data
        self.best_f1 = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        val_predict = (self.model.predict(self.validation_data[0]) > 0.5).astype(int)
        val_f1 = f1_score(self.validation_data[1], val_predict)
        logs['val_f1_score'] = val_f1
        
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.best_weights = self.model.get_weights()
            print(f"\nНовый лучший F1-score: {val_f1:.4f}")
            
            # Сохраняем модель, если F1-score >= 0.9
            if val_f1 >= 0.9:
                self.model.save("best_model.keras")
                print("Модель сохранена (F1-score >= 0.9)")

def load_and_prepare_data(file_path):
    """Загрузка и подготовка данных"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data, columns=['text', 'label'])
    
    # Преобразуем метки в числовой формат
    label_map = {'non-suicide': 0, 'suicide': 1}
    df['label'] = df['label'].map(label_map)
    
    return df

def create_model(vocab_size, max_length):
    """Создание модели"""
    model = Sequential([
        Embedding(vocab_size, 128),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    # Загрузка данных
    print("Загрузка данных...")
    df = load_and_prepare_data("RU_dataset_normalized.json")
    
    # Разделение на train и validation
    X_train, X_val, y_train, y_val = train_test_split(
        df['text'], df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    # Токенизация текста
    print("Токенизация текста...")
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    # Сохранение токенизатора
    import pickle
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    
    # Преобразование текста в последовательности
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    
    # Паддинг последовательностей
    max_length = 500  # Максимальная длина последовательности
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
    X_val_padded = pad_sequences(X_val_seq, maxlen=max_length, padding='post', truncating='post')
    
    # Создание модели
    print("Создание модели...")
    model = create_model(vocab_size=10000, max_length=max_length)
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Изменено с val_f1_score на val_loss
        patience=5,
        restore_best_weights=True,
        mode='min'
    )
    
    f1_callback = F1ScoreCallback(validation_data=(X_val_padded, y_val))
    
    # Обучение модели
    print("Обучение модели...")
    history = model.fit(
        X_train_padded, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val_padded, y_val),
        callbacks=[early_stopping, f1_callback]
    )
    
    # Оценка модели
    print("\nОценка модели на валидационных данных:")
    y_pred = (model.predict(X_val_padded) > 0.5).astype(int)
    f1 = f1_score(y_val, y_pred)
    print(f"F1-score: {f1:.4f}")
    print("\nПолный отчет о классификации:")
    print(classification_report(y_val, y_pred))
    
    # Сохранение модели
    if f1 >= 0.9:
        model.save("depression_detector.keras")
        print("\nМодель сохранена как 'depression_detector.keras' (F1-score >= 0.9)")
    else:
        print("\nМодель не достигла требуемого F1-score >= 0.9")
    
    # Построение графиков
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == "__main__":
    train_model() 