# Hugging Face Spaces Setup

This file contains instructions specific to deploying on Hugging Face Spaces.

## Files Required

| File | Description |
|------|-------------|
| `app.py` | Main Gradio application |
| `requirements_hf.txt` | Python dependencies |
| `model/hyp_param_3.pt` | Trained YOLOv8 model (tracked via Git LFS) |

## Quick Deploy

### Method 1: From GitHub

1. Go to [hf.co/new-space](https://hf.co/new-space)
2. Select **Gradio** SDK
3. Choose **Python** hardware
4. Under "Repository Settings", check "Include private git repositories"
5. Enter: `https://github.com/KevinTegar/Welding-detection`
6. Set **HF_SPACE_ID** to your desired space name
7. Create!

### Method 2: Manual Upload

1. Create a new Space at [hf.co/new-space](https://hf.co/new-space)
2. Select **Gradio** SDK
3. Clone the space repo locally
4. Copy these files:
   ```bash
   cp app.py /path/to/your-space/
   cp requirements_hf.txt /path/to/your-space/requirements.txt
   cp -r model /path/to/your-space/
   ```
5. Push to the space repo

## Hardware Selection

For YOLOv8 inference, select:
- **CPU** for basic usage (free tier)
- **GPU** (T4 small) for faster inference (paid tier)

## Environment Variables

No special environment variables required. The app uses default Gradio settings.

## Troubleshooting

### Model not loading?
Make sure `model/hyp_param_3.pt` is properly tracked in Git LFS:
```bash
git lfs install
git lfs track "*.pt"
git add model/*.pt
```

### Out of memory?
Reduce image size or lower batch size in `app.py`.

### Slow inference?
Upgrade to GPU hardware in Space settings.