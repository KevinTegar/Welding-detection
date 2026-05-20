"""
Welding Defect Detection - Gradio App for Hugging Face Spaces
=============================================================
This app uses YOLOv8 to detect welding defects in images.
"""

import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from datetime import datetime


# Load model - use LFS-tracked model from repo
MODEL_PATH = "model/hyp_param_3.pt"
OUTPUT_DIR = "static/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names

# Color scheme - Industrial theme
COLORS = {
    'primary': '#1a73e8',      # Blue
    'secondary': '#f57c00',    # Orange
    'success': '#4caf50',      # Green (Good Weld)
    'warning': '#ffc107',      # Yellow (Bad Weld)
    'danger': '#f44336',       # Red (Defect)
    'light_bg': '#ffffff',
    'dark_bg': '#1a1a1a',
}


def iou(boxA, boxB):
    """Calculate Intersection over Union between two boxes."""
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
    """Non-Maximum Suppression to filter overlapping detections."""
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


def save_output_image(img):
    """Save annotated image and return path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"welding_detection_{timestamp}.jpg"
    filepath = os.path.join(OUTPUT_DIR, filename)
    img.save(filepath, "JPEG", quality=95)
    return filepath, filename


def detect_welding(image, conf_threshold=0.25):
    """
    Detect welding defects in the uploaded image.

    Args:
        image: PIL Image or numpy array
        conf_threshold: Confidence threshold for detections

    Returns:
        Annotated image with bounding boxes
    """
    if image is None:
        return None

    # Convert to PIL if numpy array
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Run model prediction
    results = model.predict(image, conf=conf_threshold)
    filtered_boxes = improved_nms(results)

    # Draw results
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

    class_names = model.names
    color_map = ['orange', 'red', 'green']

    detection_summary = []

    for box_info in filtered_boxes:
        x1, y1, x2, y2 = [int(c) for c in box_info['box']]
        label = f"{class_names[box_info['cls']]} {box_info['conf']:.2f}"
        color_name = color_map[box_info['cls'] % len(color_map)]

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color_name, width=3)

        # Calculate text size for background
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw label background
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width, y1], fill=color_name)

        # Draw text
        draw.text((x1, y1 - text_height - 2), label, fill='white', font=font)

        detection_summary.append(f"{label}")

    return img, detection_summary


# Gradio Interface
with gr.Blocks(
    title="Deteksi Cacat Las - YOLOv8",
) as demo:
    gr.Markdown(
        """
        # 🔬 Deteksi Cacat Las - YOLOv8
        Upload gambar untuk mendeteksi cacat las menggunakan model YOLOv8 yang telah ditraining.
        """
    )

    gr.Markdown(
        """
        ### 📌 Panduan Kategori Deteksi
        - 🟩 **Good Weld**: Hasil pengelasan yang baik
        - 🟧 **Bad Weld**: Hasil pengelasan yang kurang baik
        - 🟥 **Defect**: Terdeteksi cacat yang nyata pada hasil las
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Upload Gambar",
                type="pil",
                height=400
            )
            confidence = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.25,
                step=0.05,
                label="Confidence Threshold"
            )
            detect_btn = gr.Button("🔍 Deteksi", variant="primary")

        with gr.Column():
            output_image = gr.Image(
                label="Hasil Deteksi",
                type="pil",
                height=400
            )
            detection_list = gr.JSON(
                label="Deteksi Terdeteksi"
            )

    detect_btn.click(
        fn=detect_welding,
        inputs=[input_image, confidence],
        outputs=[output_image, detection_list]
    )

    gr.Examples(
        examples=[
            # Add example images here if available
        ],
        inputs=[input_image]
    )

    gr.Markdown(
        """
        ---
        Built with ❤️ using Gradio & Ultralytics YOLOv8
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)