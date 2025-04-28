document.addEventListener('DOMContentLoaded', function () {
    const inputText = document.getElementById('inputText');
    const fileInput = document.getElementById('fileInput');
    const recognizedText = document.getElementById('recognizedText');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const modal = document.getElementById('modal');
    const verdictText = document.getElementById('verdictText');
    const okBtn = document.getElementById('okBtn');
    const closeModal = document.getElementById('closeModal');

    // Отображение текста из textarea в recognizedText
    inputText.addEventListener('input', function () {
        recognizedText.textContent = inputText.value;
    });

    // Загрузка текста из файла
    fileInput.addEventListener('change', function (e) {
        const file = e.target.files[0];
        if (file && file.type === "text/plain") {
            const reader = new FileReader();
            reader.onload = function (evt) {
                inputText.value = evt.target.result;
                recognizedText.textContent = evt.target.result;
            };
            reader.readAsText(file, 'UTF-8');
        } else {
            alert('Пожалуйста, выберите текстовый файл (.txt)');
        }
    });

    // Анализ текста
    analyzeBtn.addEventListener('click', function () {
        const text = inputText.value.trim();
        if (!text) {
            alert('Введите или загрузите текст для анализа!');
            return;
        }
        fetch('/analyze', {
            method: 'POST',
            body: new URLSearchParams({ text }),
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                verdictText.textContent = data.error;
            } else {
                verdictText.textContent = `Вердикт: ${data.verdict}\n(Уверенность: ${(data.prob * 100).toFixed(1)}%)`;
            }
            modal.style.display = 'block';
        })
        .catch(() => {
            verdictText.textContent = 'Ошибка при анализе текста.';
            modal.style.display = 'block';
        });
    });

    // Закрытие модального окна
    okBtn.addEventListener('click', function () {
        modal.style.display = 'none';
    });
    closeModal.addEventListener('click', function () {
        modal.style.display = 'none';
    });
    window.onclick = function(event) {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    };
});