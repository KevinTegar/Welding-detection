# Welding Detection UI - Gradio Redesign Specification

**Date:** 2026-05-20
**Author:** Claude (with user approval)
**Status:** Approved

---

## Overview

Redesign the Gradio interface for the Welding Defect Detection app with an industrial/technical theme, vertical stack layout, and enhanced features.

---

## Design Specifications

### Theme: Industrial/Teknik

| Element | Color |
|---------|-------|
| Primary | Blue (#1a73e8) |
| Secondary | Orange (#f57c00) |
| Accent Success | Green (#4caf50) |
| Accent Warning | Yellow (#ffc107) |
| Accent Danger | Red (#f44336) |
| Background (Light) | White (#ffffff) |
| Background (Dark) | Dark Gray (#1a1a1a) |
| Text (Light) | Dark Gray (#333333) |
| Text (Dark) | Light Gray (#f5f5f5) |

### Layout: Vertical Stack

```
1. Header (Logo + Title)
2. Legend (Category Guide)
3. Theme Toggle (Light/Dark)
4. Upload Section
5. Controls (Confidence Slider + Detect Button)
6. Results Section
7. History/Gallery
8. Footer (Model Info)
```

---

## Features

### 1. Upload Section
- Drag & drop area
- File picker button
- Image preview after upload
- Supported formats: jpg, jpeg, png, bmp

### 2. Confidence Slider
- Range: 0.1 to 1.0
- Default: 0.25
- Step: 0.05
- Label showing current value

### 3. Detect Button
- Primary style (blue background)
- Icon: 🔍
- Loading state during inference

### 4. Results Section
- Annotated image with bounding boxes
- Color coding:
  - 🟩 Good Weld: Green
  - 🟧 Bad Weld: Orange
  - 🟥 Defect: Red
- Detection summary list
- Download button (💾)

### 5. Download Feature
- Downloads annotated image as JPG
- Filename: `welding_detection_YYYYMMDD_HHMMSS.jpg`

### 6. History/Gallery
- Stores up to 10 recent detections
- Thumbnail grid view
- Click to view full result
- Persists in session (not database)

### 7. Light/Dark Toggle
- Toggle button in header
- Smooth transition
- Persists preference in session

### 8. Category Legend
- Displayed at top
- Color + description for each class

### 9. Model Info Footer
- Model name (YOLOv8)
- Number of classes
- Confidence threshold info

---

## Component Mapping

| Component | Gradio Element |
|-----------|---------------|
| Header | gr.Markdown |
| Legend | gr.Markdown (styled) |
| Theme Toggle | gr.Button (toggle behavior) |
| Upload | gr.Image (type="filepath") |
| Confidence | gr.Slider |
| Detect Button | gr.Button |
| Results Image | gr.Image |
| Detection List | gr.JSON |
| Download Button | gr.Button |
| History Gallery | gr.Gallery |
| Footer | gr.Markdown |

---

## State Management

```python
# Session state variables
- theme: "light" | "dark" (default: "light")
- history: list of {image, detections, timestamp}
- current_result: dict with annotated image path
```

---

## File Output

- Output path: `static/outputs/`
- Filename pattern: `welding_detection_{timestamp}.jpg`
- Auto-cleanup: Keep last 50 files

---

## Acceptance Criteria

1. ✅ App loads with industrial blue/orange theme
2. ✅ Vertical layout works on mobile and desktop
3. ✅ Light/Dark toggle switches theme smoothly
4. ✅ Image upload via drag-drop or file picker
5. ✅ Confidence slider adjustable (0.1-1.0)
6. ✅ Detection displays bounding boxes with correct colors
7. ✅ Download button saves annotated image
8. ✅ History gallery shows last 10 detections
9. ✅ Legend clearly shows category colors
10. ✅ Footer displays model info

---

## Implementation Priority

1. Basic layout structure
2. Theme toggle
3. Upload + detection
4. Results display
5. Download feature
6. History gallery
7. Polish and styling