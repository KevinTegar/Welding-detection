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

THEME_JS = """
<script>
function toggleTheme() {
    var body = document.querySelector('body');
    var header = document.querySelector('#welding-header');
    var legend = document.querySelector('#welding-legend');
    var footer = document.querySelector('#welding-footer');

    if (body.style.backgroundColor === 'rgb(26, 26, 26)') {
        // Switch to light
        body.style.backgroundColor = '#ffffff';
        body.style.color = '#333333';
        if (header) { header.style.background = 'linear-gradient(135deg, #1a73e8, #0d47a1)'; header.style.color = 'white'; }
        if (legend) { legend.style.background = '#f5f5f5'; legend.style.color = '#333333'; }
        if (footer) { footer.style.background = '#f5f5f5'; footer.style.color = '#333333'; }
    } else {
        // Switch to dark
        body.style.backgroundColor = '#1a1a1a';
        body.style.color = '#f5f5f5';
        if (header) { header.style.background = 'linear-gradient(135deg, #1565c0, #0d47a1)'; header.style.color = 'white'; }
        if (legend) { legend.style.background = '#2d2d2d'; legend.style.color = '#f5f5f5'; }
        if (footer) { footer.style.background = '#2d2d2d'; footer.style.color = '#f5f5f5'; }
    }
}
</script>
"""

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
    Detect welding defects with bounding boxes and save output.

    Args:
        image: PIL Image or numpy array
        conf_threshold: Confidence threshold for detections

    Returns:
        tuple: (annotated_image, detection_summary_list, output_filepath)
    """
    if image is None:
        return None, [], None

    # Convert to PIL if numpy array
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Run model prediction
    results = model.predict(image, conf=conf_threshold)
    filtered_boxes = improved_nms(results)

    # Draw results
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)

    # Font setup
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()

    # Color mapping: index -> (name, RGB)
    # 0=Bad Weld (Yellow), 1=Defect (Red), 2=Good Weld (Green)
    color_map = [
        ('#ffc107', (255, 193, 7)),   # Bad Weld - Yellow
        ('#f44336', (244, 67, 54)),  # Defect - Red
        ('#4caf50', (76, 175, 80)),  # Good Weld - Green
    ]

    detection_summary = []

    for box_info in filtered_boxes:
        x1, y1, x2, y2 = [int(c) for c in box_info['box']]
        cls_idx = box_info['cls'] % len(color_map)
        color_name, color_rgb = color_map[cls_idx]
        label = f"{CLASS_NAMES[box_info['cls']]} {box_info['conf']:.2f}"

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color_name, width=3)

        # Calculate text size for background
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw label background
        draw.rectangle([x1, y1 - text_height - 6, x1 + text_width + 4, y1], fill=color_name)
        draw.text((x1 + 2, y1 - text_height - 4), label, fill='white', font=font)

        detection_summary.append({
            'class': CLASS_NAMES[box_info['cls']],
            'confidence': round(float(box_info['conf']), 2),
            'bbox': [int(c) for c in box_info['box']]
        })

    # Save output image
    output_path, output_filename = save_output_image(img)

    return img, detection_summary, output_path


def build_header():
    return gr.Markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1a73e8, #0d47a1); color: white; border-radius: 12px; margin-bottom: 20px;">
        <h1 style="margin: 0;">🔬 Deteksi Cacat Las - YOLOv8</h1>
        <p style="margin: 10px 0 0 0; opacity: 0.9;">YOLOv8 trained model for welding defect detection</p>
    </div>
    """, elem_id="welding-header")


def build_legend():
    return gr.Markdown("""
    <div style="display: flex; justify-content: center; gap: 30px; padding: 15px; background: #f5f5f5; border-radius: 8px; margin-bottom: 20px; flex-wrap: wrap;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="background: #4caf50; color: white; padding: 4px 12px; border-radius: 4px; font-weight: 600;">Good Weld</span>
            <span>Hasil pengelasan yang baik</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="background: #ffc107; color: white; padding: 4px 12px; border-radius: 4px; font-weight: 600;">Bad Weld</span>
            <span>Hasil pengelasan yang kurang baik</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="background: #f44336; color: white; padding: 4px 12px; border-radius: 4px; font-weight: 600;">Defect</span>
            <span>Terdeteksi cacat yang nyata</span>
        </div>
    </div>
    """, elem_id="welding-legend")


def build_theme_toggle():
    return gr.Button(
        "🌙 Dark Mode",
        size="sm",
        variant="secondary",
        elem_id="theme-toggle"
    )


def build_footer():
    return gr.Markdown(
        """
        ---
        Built with ❤️ using Gradio & Ultralytics YOLOv8
        """,
        elem_id="welding-footer"
    )


# Gradio Interface
with gr.Blocks(
    title="Deteksi Cacat Las - YOLOv8",
    theme=gr.themes.Soft(),
    js=THEME_JS
) as demo:
    # Header
    build_header()

    # Theme toggle row
    with gr.Row():
        build_theme_toggle()

    # Legend
    build_legend()

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

    # Footer
    build_footer()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)