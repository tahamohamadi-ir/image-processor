
***

## `Documents/QUICK_REFERENCE.md`

```markdown
# Image Processor API — Quick Reference

> Fast lookup card. For full context see TECHNICAL_DOCS.md.

---

## Commands

```bash
# Install
pip install -r requirements.txt

# Run server
python -m app.main
# → http://localhost:8000/docs

# Run tests
pytest tests/ -v

# Docker
docker compose up --build

# Optional: deep denoiser (NAFNet weights)
# Download NAFNet-SIDD-width64.pth → place in models/

# Optional: Real-ESRGAN (if basicsr needed)
pip install basicsr --no-build-isolation
pip install realesrgan --no-build-isolation
```
Pipeline Order
```text
Input (RGB) →  Denoise →  Shadow →  BG Remove →  Enhance → Output (RGBA)[3][4][1][2]
All steps optional. Steps 1-3 on by default. Step 4 off by default.
```
Processor Fallback Chains

| Processor      | Best                 | Fallback             | Always-on      |
| -------------- | -------------------- | -------------------- | -------------- |
| Denoiser       | NAFNet (manual DL)   | —                    | OpenCV NLM     |
| Shadow Remover | ShadowFormer (TODO)  | —                    | OpenCV Retinex |
| BG Remover     | BiRefNet + PyMatting | BiRefNet + Bilateral | —              |
| Enhancer       | super-image A2N/EDSR | Real-ESRGAN          | PIL LANCZOS    |

Config .env Keys
text
DEVICE=cuda             # cuda | cpu
USE_FP16=true
BIREFNET_INPUT_SIZE=1024  # 512 | 1024 | 2048
BIREFNET_REFINE_MATTING=true
DENOISE_STRENGTH=0.5    # 0.0-1.0
ENHANCEMENT_MODEL=a2n   # a2n | edsr
REALESRGAN_SCALE=2      # 2 | 4
MAX_IMAGE_SIZE=4096
MAX_FILE_SIZE_MB=50
CORS_ORIGINS=["*"]      # must be JSON array
DEBUG=false
API Endpoints

| Method | Path            | Body                     | Returns            |
| ------ | --------------- | ------------------------ | ------------------ |
| GET    | /health         | —                        | JSON status        |
| GET    | /api/v1/models  | —                        | JSON loaded models |
| POST   | /api/v1/process | multipart: file + config | Binary image       |

ProcessingConfig Defaults
json
{
  "denoise": true,
  "remove_shadow": true,
  "remove_background": true,
  "enhance": false,
  "enhance_scale": 2,
  "use_matting": true,
  "output_format": "png"
}
Model Registry Keys

| Key            | Loaded by                  |
| -------------- | -------------------------- |
| denoiser       | DenoiseProcessor           |
| shadow_remover | ShadowRemovalProcessor     |
| birefnet       | BackgroundRemovalProcessor |
| enhancer       | EnhancementProcessor       |

Common Fixes

| Error                        | Fix                                      |
| ---------------------------- | ---------------------------------------- |
| basicsr KeyError __version__ | pip install basicsr --no-build-isolation |
| CUDA out of memory           | USE_FP16=true or DEVICE=cpu              |
| BiRefNet download fails      | Check internet / HuggingFace access      |
| CORS_ORIGINS parse error     | Set as JSON: CORS_ORIGINS=["*"]          |
| Output all transparent       | BiRefNet confidence low — check input    |

VRAM Budget (FP16)

| Model    | VRAM   |
| -------- | ------ |
| BiRefNet | ~3.5GB |
| NAFNet   | ~0.5GB |
| A2N x2   | ~0.3GB |
| Total    | ~4.3GB |

Add New Processor (5 steps)
app/processors/my_processor.py — extend BaseProcessor

app/processors/__init__.py — add import

app/api/schemas.py — add bool field to ProcessingConfig

app/core/pipeline.py — add if config.my_step: block

tests/test_my_processor.py — add tests













