from flask import Flask, render_template, request, jsonify
import os
from depres_model.predict import load_model_and_tokenizer, predict_text

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
    prediction = predict_text(text, model, tokenizer)
    verdict = 'Депрессивный' if prediction > 0.5 else 'Не депрессивный'
    return jsonify({'verdict': verdict, 'prob': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)