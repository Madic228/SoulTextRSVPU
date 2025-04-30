# SoulTextRSVPU

**SoulTextRSVPU** — это веб-приложение на Flask для анализа депрессивности текста с помощью нейронной сети. Приложение поддерживает загрузку текстовых файлов (txt, doc, docx, pdf), очистку текста, предобработку и предсказание вероятности депрессивного содержания.

---

## Возможности

- Веб-интерфейс для анализа текста и файлов.
- Поддержка форматов: `.txt`, `.doc`, `.docx`, `.pdf`.
- Очистка и нормализация текста.
- Предсказание депрессивности текста с помощью обученной нейронной сети (TensorFlow/Keras).
- Просмотр статистики по датасету и обучение модели.
- Поддержка работы с GPU (если доступно).

---

## Быстрый старт

### 1. Клонируйте репозиторий

```sh
git clone https://github.com/yourusername/SoulTextRSVPU.git
cd SoulTextRSVPU
```

### 2. Установите зависимости

Рекомендуется использовать виртуальное окружение:

```sh
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Пример содержимого `requirements.txt`:**
```
Flask
python-docx
PyPDF2
numpy
pandas
tqdm
tensorflow
scikit-learn
matplotlib
```

### 3. Подготовьте модель

- Обучите модель с помощью `server/depres_model/train_model.py` или используйте уже обученные веса (`depression_detector_weights.h5`) и токенизатор (`tokenizer.pkl`).
- Для обучения используйте нормализованный датасет (`RU_dataset_normalized.json`).

### 4. Запустите сервер

```sh
python server/app.py
```

Сайт будет доступен по адресу: [http://localhost:5000](http://localhost:5000)

---

## Быстрый старт через Docker

### 1. Соберите Docker-образ

```sh
docker build -t soultextrsvpu .
```

### 2. Запустите контейнер

```sh
docker run -p 5000:5000 soultextrsvpu
```

### 3. Откройте сайт

Перейдите в браузере по адресу: [http://localhost:5000](http://localhost:5000)

---

**Примечание:**  
Перед сборкой Docker-образа убедитесь, что в проекте есть файл `requirements.txt` и все необходимые веса/токенизатор в папке `server/depres_model/`.

---

## Структура проекта

```
server/
│
├── app.py                      # Flask-приложение
├── depres_model/
│   ├── predict.py              # Загрузка модели и предсказание
│   ├── train_model.py          # Обучение модели
│   ├── normalize_dataset.py    # Очистка и нормализация датасета
│   ├── RU_dataset_normalized.json  # Пример нормализованного датасета
│   ├── analyze_dataset.py      # Анализ датасета
│   ├── depression_detector_weights.h5  # Веса модели
│   └── tokenizer.pkl           # Токенизатор
│
├── static/
│   ├── style.css               # Стили
│   ├── script.js               # JS-логика
│   └── cross-small-svgrepo-com.svg # SVG для кнопки отмены
│
└── templates/
    └── main.html               # Основная страница


```

---

## Использование

1. Введите текст вручную или загрузите файл.
2. При необходимости очистите или измените текст.
3. Нажмите кнопку "Определить степень депрессивности".
4. Результат появится в модальном окне.

---

## Обработка файлов

- Для `.txt` — текст читается напрямую.
- Для `.docx` и `.pdf` — текст извлекается на сервере.
- Для других форматов поддержка может быть добавлена при необходимости.

---

## Обучение модели

Процесс нашего обучения и тестирования: DepresURFU.ipynb (https://github.com/Madic228/SoulTextRSVPU/blob/master/DepresURFU.ipynb)

Если захотите обучить сами, используйте:

```sh
python server/depres_model/train_model.py
```

Параметры и структура модели можно изменить в соответствующем файле.

---

## Примечания

- Для работы с большими моделями рекомендуется использовать GPU.
- Для публичного доступа используйте reverse proxy или tunneling (например, Cloudflare Tunnel, ngrok).
- Не используйте встроенный Flask-сервер для продакшена.

---

## Лицензия

MIT License

---

## Авторы

- Madic228      (https://github.com/Madic228)
- Egor-Ulanov   (https://github.com/Egor-Ulanov)
- Cripochec     (https://github.com/Cripochec)
