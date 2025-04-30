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
    fileInput.addEventListener('change', async function(e) {
        const file = e.target.files[0];
        if (!file) return;

        const ext = file.name.split('.').pop().toLowerCase();
        if (ext === "txt") {
            const text = await file.text();
            cleanAndShowText(text);
        } else if (["doc", "docx", "pdf"].includes(ext)) {
            const formData = new FormData();
            formData.append('file', file);
            fetch('/extract_text', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                if (data.text) {
                    cleanAndShowText(data.text);
                } else {
                    recognizedText.innerText = data.error || "Ошибка при извлечении текста.";
                }
            });
        } else {
            recognizedText.innerText = "Неподдерживаемый тип файла.";
        }
    });

    document.getElementById('fileInput').addEventListener('change', function() {
        const label = document.querySelector('.custom-file-label');
        if (this.files.length > 0) {
            label.textContent = this.files[0].name;
        } else {
            label.textContent = 'Выбрать файл';
        }
    });

    document.getElementById('clearFileBtn').addEventListener('click', function() {
        document.getElementById('fileInput').value = "";
        document.querySelector('.custom-file-label').textContent = 'Выбрать файл';
        recognizedText.innerText = "";
    });

    // Функция для отправки текста на сервер для очистки и отображения
    function cleanAndShowText(text) {
        fetch('/clean_text', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text})
        })
        .then(res => res.json())
        .then(data => {
            recognizedText.innerText = data.cleaned_text;
        });
    }

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