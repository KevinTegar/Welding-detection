from flask import Flask, request, render_template, jsonify, url_for
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import uuid
import numpy as np

app = Flask(__name__)
model = YOLO("model/hyp_param_3.pt")
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def improved_nms(results, iou_threshold=0.5):
    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy()

    indices = np.argsort(-conf)
    selected = []

    for idx in indices:
        keep = True
        for kept in selected:
            iou_score = iou(xyxy[idx], xyxy[kept])
            if iou_score > iou_threshold:
                keep = False
                break
        if keep:
            selected.append(idx)

    final = []
    for idx in selected:
        final.append({
            'box': xyxy[idx],
            'conf': conf[idx],
            'cls': int(cls[idx])
        })
    return final

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    # Run model prediction
    results = model.predict(image_path, conf=0.25)
    filtered_boxes = improved_nms(results)

    # Draw results
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    class_names = model.names
    color_map = ['orange', 'red', 'green']

    for box_info in filtered_boxes:
        x1, y1, x2, y2 = [int(c) for c in box_info['box']]
        label = f"{class_names[box_info['cls']]} {box_info['conf']:.2f}"
        color_name = color_map[box_info['cls'] % len(color_map)]

        # Gambar bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color_name, width=3)

        # Hitung ukuran teks untuk background
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Gambar background label
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width, y1], fill=color_name)
        
        # Gambar teks dengan warna putih
        draw.text((x1, y1 - text_height - 2), label, fill='white', font=font)

    filtered_path = os.path.join(UPLOAD_FOLDER, f"filtered_{filename}")
    img.save(filtered_path)

    web_image_path = url_for('static', filename=f"uploads/filtered_{filename}")
    return jsonify({"image_url": web_image_path})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
