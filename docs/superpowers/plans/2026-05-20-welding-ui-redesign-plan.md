# Welding Detection UI - Gradio Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign Gradio interface with industrial theme, vertical layout, and enhanced features (download, history, theme toggle)

**Architecture:** Single-file Gradio app with session state management. Theme handled via CSS injection. History stored in Gradio session state. Output images saved to static directory.

**Tech Stack:** Gradio 6.x, Ultralytics YOLOv8, PIL, Python standard libraries

---

## File Structure

| File | Purpose |
|------|---------|
| `app.py` | Main Gradio application (complete rewrite) |
| `static/outputs/` | Directory for annotated output images |

---

## Tasks

### Task 1: Create Output Directory and Helper Functions

**Files:**
- Modify: `app.py` (add imports and helper functions)

- [ ] **Step 1: Write imports and constants**

```python
import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from datetime import datetime

# Constants
MODEL_PATH = "model/hyp_param_3.pt"
OUTPUT_DIR = "static/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
```

- [ ] **Step 2: Write theme CSS**

```python
# Theme CSS templates
LIGHT_THEME_CSS = """
.gradio-container { --primary-color: #1a73e8; }
.welding-header { text-align: center; padding: 20px; background: linear-gradient(135deg, #1a73e8, #0d47a1); color: white; border-radius: 12px; margin-bottom: 20px; }
.welding-legend { display: flex; justify-content: center; gap: 20px; padding: 15px; background: #f5f5f5; border-radius: 8px; margin-bottom: 20px; }
.legend-item { display: flex; align-items: center; gap: 8px; font-weight: 600; }
"""

DARK_THEME_CSS = """
.gradio-container { --primary-color: #64b5f6; --secondary-color: #ffb74d; }
body { background-color: #1a1a1a; color: #f5f5f5; }
.welding-header { text-align: center; padding: 20px; background: linear-gradient(135deg, #1565c0, #0d47a1); color: white; border-radius: 12px; margin-bottom: 20px; }
.welding-legend { display: flex; justify-content: center; gap: 20px; padding: 15px; background: #2d2d2d; border-radius: 8px; margin-bottom: 20px; }
"""
```

- [ ] **Step 3: Write model loading and detection functions**

```python
# Load model once at startup
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names

def iou(boxA, boxB):
    """Calculate Intersection over Union between two boxes."""
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
            if iou(xyxy[idx], xyxy[kept]) > iou_threshold:
                keep = False
                break
        if keep:
            selected.append(idx)
    return [{'box': xyxy[i], 'conf': conf[i], 'cls': int(cls[i])} for i in selected]
```

- [ ] **Step 4: Write save_output_image function**

```python
def save_output_image(img):
    """Save annotated image and return path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"welding_detection_{timestamp}.jpg"
    filepath = os.path.join(OUTPUT_DIR, filename)
    img.save(filepath, "JPEG", quality=95)
    return filepath, filename
```

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: add helper functions, theme constants, and output directory"
```

---

### Task 2: Write Core Detection Function

**Files:**
- Modify: `app.py` (update detect_welding function)

- [ ] **Step 1: Write enhanced detect_welding function**

```python
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
    color_map = [
        ('#ffc107', (255, 193, 7)),   # Bad Weld - Yellow
        ('#f44336', (244, 67, 54)),    # Defect - Red
        ('#4caf50', (76, 175, 80)),    # Good Weld - Green
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
```

- [ ] **Step 2: Test function exists**

Run: `python -c "from app import detect_welding; print('detect_welding OK')"`
Expected: `detect_welding OK`

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: implement detection function with bounding box drawing"
```

---

### Task 3: Build Gradio UI - Header and Legend

**Files:**
- Modify: `app.py` (add Gradio interface building)

- [ ] **Step 1: Write CSS and JavaScript for theme toggle**

```python
THEME_JS = """
<script>
function toggleTheme() {
    var body = document.querySelector('body');
    var container = document.querySelector('.gradio-container');
    if (body.style.backgroundColor === 'rgb(26, 26, 26)') {
        body.style.backgroundColor = '#ffffff';
        body.style.color = '#333333';
    } else {
        body.style.backgroundColor = '#1a1a1a';
        body.style.color = '#f5f5f5';
    }
}
</script>
"""
```

- [ ] **Step 2: Write header and legend components**

```python
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
```

- [ ] **Step 3: Write theme toggle button**

```python
def build_theme_toggle():
    return gr.Button(
        "🌙 Dark Mode",
        size="sm",
        variant="secondary",
        elem_id="theme-toggle"
    )
```

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add header, legend, and theme toggle components"
```

---

### Task 4: Build Gradio UI - Upload and Controls

**Files:**
- Modify: `app.py` (add upload section)

- [ ] **Step 1: Write upload section**

```python
def build_upload_section():
    with gr.Column():
        input_image = gr.Image(
            label="📤 Upload Gambar",
            type="pil",
            height=350,
            elem_id="input-image"
        )

        with gr.Row():
            confidence = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.25,
                step=0.05,
                label="Confidence Threshold",
                elem_id="confidence-slider"
            )
            detect_btn = gr.Button(
                "🔍 Deteksi",
                variant="primary",
                size="lg",
                elem_id="detect-btn"
            )
```

- [ ] **Step 2: Commit**

```bash
git add app.py
git commit -m "feat: add upload section with image input and confidence slider"
```

---

### Task 5: Build Gradio UI - Results Section

**Files:**
- Modify: `app.py` (add results display)

- [ ] **Step 1: Write results section with download**

```python
def build_results_section():
    with gr.Column():
        output_image = gr.Image(
            label="📊 Hasil Deteksi",
            type="pil",
            height=350,
            elem_id="output-image"
        )

        with gr.Row():
            detection_list = gr.JSON(
                label="📋 Detail Deteksi",
                elem_id="detection-list"
            )
            download_btn = gr.Button(
                "💾 Download Hasil",
                variant="secondary",
                size="sm",
                elem_id="download-btn"
            )
```

- [ ] **Step 2: Write download handler**

```python
def download_result(filepath):
    """Return file for download."""
    if filepath and os.path.exists(filepath):
        return filepath
    return None
```

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add results section with download functionality"
```

---

### Task 6: Build Gradio UI - History Gallery

**Files:**
- Modify: `app.py` (add history gallery)

- [ ] **Step 1: Write history gallery section**

```python
def build_history_section():
    return gr.Gallery(
        label="📁 Riwayat Deteksi",
        columns=5,
        rows=2,
        object_fit="contain",
        height="auto",
        elem_id="history-gallery"
    )
```

- [ ] **Step 2: Write history update function**

```python
def update_history(history, new_result):
    """Add new result to history (keep last 10)."""
    if new_result is None:
        return history
    history = history or []
    history.insert(0, new_result)
    return history[:10]  # Keep only last 10
```

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add history gallery section"
```

---

### Task 7: Build Gradio UI - Footer and Assemble

**Files:**
- Modify: `app.py` (complete assembly)

- [ ] **Step 1: Write footer**

```python
def build_footer():
    return gr.Markdown("""
    <div style="text-align: center; padding: 15px; background: #f5f5f5; border-radius: 8px; margin-top: 20px;">
        <p style="margin: 0; color: #666;">
            <strong>Model:</strong> YOLOv8 | <strong>Classes:</strong> 3 (Good Weld, Bad Weld, Defect) | <strong>Confidence:</strong> 0.25 (default)
        </p>
        <p style="margin: 10px 0 0 0; font-size: 12px; color: #999;">
            Built with ❤️ using Gradio & Ultralytics
        </p>
    </div>
    """, elem_id="welding-footer")
```

- [ ] **Step 2: Assemble complete Gradio interface**

```python
# Complete Gradio Interface
with gr.Blocks(
    title="Deteksi Cacat Las - YOLOv8",
    head=THEME_JS,
) as demo:

    # Header
    build_header()

    # Legend
    build_legend()

    # Theme Toggle
    theme_btn = build_theme_toggle()

    # Upload Section
    with gr.Column():
        input_image = gr.Image(label="📤 Upload Gambar", type="pil", height=350)
        with gr.Row():
            confidence = gr.Slider(minimum=0.1, maximum=1.0, value=0.25, step=0.05, label="Confidence Threshold")
            detect_btn = gr.Button("🔍 Deteksi", variant="primary", size="lg")

    # Results Section
    with gr.Column():
        output_image = gr.Image(label="📊 Hasil Deteksi", type="pil", height=350)
        with gr.Row():
            detection_list = gr.JSON(label="📋 Detail Deteksi")
            download_btn = gr.Button("💾 Download Hasil", variant="secondary")

    # History Gallery
    history_gallery = build_history_section()

    # Footer
    build_footer()

    # Event Handlers
    detect_btn.click(
        fn=detect_welding,
        inputs=[input_image, confidence],
        outputs=[output_image, detection_list]
    )

    # Download handler (output file path from detection result)
    download_btn.click(
        fn=lambda x: x if x else None,
        inputs=[output_image],
        outputs=[]
    )
```

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: assemble complete Gradio interface with all sections"
```

---

### Task 8: Add Theme Toggle JavaScript

**Files:**
- Modify: `app.py` (add theme toggle functionality)

- [ ] **Step 1: Add JavaScript for theme toggle**

```python
THEME_JS = """
<script>
function toggleTheme() {
    var body = document.body;
    var header = document.querySelector('#welding-header');
    var legend = document.querySelector('#welding-legend');
    var footer = document.querySelector('#welding-footer');
    var btn = document.querySelector('#theme-toggle button, #theme-toggle');

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
```

- [ ] **Step 2: Wire up theme toggle button**

```python
theme_btn.click(fn=None, js="toggleTheme()")
```

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add theme toggle JavaScript functionality"
```

---

### Task 9: Test and Final Polish

**Files:**
- Modify: `app.py` (polish and test)

- [ ] **Step 1: Run app and verify all components load**

Run: `python app.py`
Expected: App starts without errors, Gradio interface accessible at localhost:7860

- [ ] **Step 2: Test upload and detection**

1. Open browser to localhost:7860
2. Upload a test image
3. Click "Deteksi" button
4. Verify:
   - Bounding boxes appear
   - Detection list shows results
   - Output image displays correctly

- [ ] **Step 3: Test theme toggle**

Click theme toggle button and verify colors change.

- [ ] **Step 4: Final commit**

```bash
git add app.py
git commit -m "chore: final polish and testing"
```

---

### Task 10: Push and Prepare for Hugging Face Spaces

**Files:**
- Modify: `app.py`, `.gitignore`

- [ ] **Step 1: Update .gitignore for outputs**

Add to `.gitignore`:
```
static/outputs/*
!static/outputs/.gitkeep
```

- [ ] **Step 2: Create .gitkeep for outputs directory**

Create empty file `static/outputs/.gitkeep`

- [ ] **Step 3: Ensure model file is in git**

Verify model is tracked with LFS:
```bash
git lfs ls-files
```
Expected: `model/hyp_param_3.pt`

- [ ] **Step 4: Push to remote**

```bash
git push origin main
```

- [ ] **Step 5: Commit final changes**

```bash
git add .gitignore static/outputs/.gitkeep
git commit -m "chore: prepare for Hugging Face Spaces deployment"
git push origin main
```

---

## Acceptance Criteria Checklist

- [ ] Industrial theme (blue/orange) applied
- [ ] Vertical layout works on mobile and desktop
- [ ] Light/Dark toggle switches theme
- [ ] Image upload via file picker
- [ ] Confidence slider adjustable (0.1-1.0)
- [ ] Detection displays bounding boxes with correct colors
- [ ] Legend shows category colors (Green/Yellow/Red)
- [ ] Model info footer displays correctly

---

## Notes

- Output images saved to `static/outputs/` with timestamp filename
- History kept in Gradio session state (resets on restart)
- Font fallback handles both Linux (DejaVu) and Windows (Arial)
- Colors match Indonesian safety/welding standards

---

**Plan complete!** Ready for implementation.