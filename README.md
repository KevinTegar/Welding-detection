# Welding Detection - YOLOv8 🕵️‍♂️

> **Indonesian**: Deteksi Cacat Las menggunakan model YOLOv8 yang telah ditraining untuk mendeteksi cacat pada hasil pengelasan.

A web application for detecting welding defects using YOLOv8 computer vision model. Upload an image and get instant detection results with bounding boxes.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)

---

## 🎯 Features

- **Real-time Detection**: Upload images and get instant welding defect detection
- **Bounding Box Visualization**: Clear visual indicators for different defect types
- **Confidence Threshold**: Adjustable sensitivity for detections
- **Three Categories**:
  - 🟩 **Good Weld**: Good welding result
  - 🟧 **Bad Weld**: Substandard welding
  - 🟥 **Defect**: Detected actual defect

---

## 🔧 Tech Stack

- **Model**: YOLOv8 (Ultralytics)
- **Framework**: Gradio
- **Backend**: PyTorch

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/KevinTegar/Welding-detection.git
cd Welding-detection

# Install dependencies
pip install -r requirements_hf.txt

# Run the app
python app.py
```

---

## 🚀 Deployment

### Hugging Face Spaces (Recommended)

1. Go to [Hugging Face Spaces](https://huggingface.co/new-space)
2. Create a new Space with **Gradio** SDK
3. Select **Python** as the hardware
4. Upload the following files:
   - `app.py`
   - `requirements_hf.txt`
   - `model/hyp_param_3.pt`
   - `README.md`
5. Click **Create Space** and wait for deployment

Your app will be available at `https://username-welding-detection.hf.space`

### Local Run

```bash
# Install requirements
pip install -r requirements_hf.txt

# Run Gradio app
python app.py
```

---

## 📊 Model Information

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8 |
| Input Size | 640x640 |
| Confidence Threshold | 0.25 (default) |
| IoU Threshold | 0.5 |

---

## 📁 Project Structure

```
Welding-detection/
├── app.py              # Main Gradio application
├── model/
│   └── hyp_param_3.pt  # Trained YOLOv8 model
├── requirements_hf.txt # Dependencies for HF Spaces
└── README.md           # This file
```

---

## 📜 License

MIT License

---

## 👤 Author

**Kevin Tegar** - [GitHub](https://github.com/KevinTegar)