from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from io import BytesIO
from docx import Document
import PyPDF2
from depres_model.predict import load_model_and_tokenizer, predict_text
from depres_model.normalize_dataset import clean_text

ALLOWED_EXTENSIONS = {'txt', 'doc', 'docx', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)

# Загрузка модели и токенизатора один раз при старте
model, tokenizer = load_model_and_tokenizer()

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text')
    if not text:
        return jsonify({'error': 'Нет текста для анализа'}), 400
    text = clean_text(text)
    prediction = predict_text(text, model, tokenizer)
    verdict = 'Депрессивный' if prediction > 0.5 else 'Не депрессивный'
    prob = float(prediction)
    if verdict == 'Не депрессивный':
        prob = 100 - prob
    return jsonify({'verdict': verdict, 'prob': prob})

    # return jsonify({'verdict': 'Не депрессивный', 'prob': 80.0})

@app.route('/clean_text', methods=['POST'])
def clean_text_api():
    data = request.get_json()
    text = data.get('text', '')
    cleaned = clean_text(text)
    return jsonify({'cleaned_text': cleaned})

@app.route('/extract_text', methods=['POST'])
def extract_text():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Неподдерживаемый тип файла'}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    text = ""
    if ext == 'txt':
        text = file.read().decode('utf-8', errors='ignore')
    elif ext == 'docx':
        doc = Document(BytesIO(file.read()))
        text = "\n".join([p.text for p in doc.paragraphs])
    elif ext == 'pdf':
        reader = PyPDF2.PdfReader(BytesIO(file.read()))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    else:
        return jsonify({'error': 'Этот формат поддерживается только для docx, pdf, txt'}), 400

    return jsonify({'text': text})

if __name__ == '__main__':
    app.run(debug=True)