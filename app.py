from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image
import os

UPLOAD_FOLDER = 'uploads'

# Загружаем модель YOLO
model = YOLO("runs/detect/train2/weights/best.pt")


# Создаем приложение Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Главная страница
@app.route('/')
def index():
    return render_template('index.html')

# API для предсказания
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    # Сохраняем загруженный файл
    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Прогон изображения через модель
    results = model.predict(source=file_path, conf = 0.2)

    # Обработка результатов
    result_summary = {
        "path_detected": False,
        "boxes": []
    }

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        confidence = float(box.conf[0])
        coordinates = box.xyxy[0].tolist()

        if label == "path":
            result_summary["path_detected"] = True

        result_summary["boxes"].append({
            "label": label,
            "confidence": confidence,
            "coordinates": coordinates
        })

    return jsonify(result_summary)

if __name__ == '__main__':
    app.run(debug=True)