# Image Processor API

FastAPI-based image processing pipeline:
**Denoise → Shadow Removal → Background Removal → Enhancement**

## Stack

| Step | Technology |
|------|-----------|
| Denoising | NAFNet (optional) / OpenCV |
| Shadow Removal | OpenCV Multi-Scale Retinex |
| Background Removal | BiRefNet + PyMatting |
| Enhancement | super-image (A2N/EDSR) |
| API | FastAPI + Uvicorn |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and edit environment file
copy .env .env.local

# 3. Run server
python -m app.main
# → http://localhost:8000/docs
