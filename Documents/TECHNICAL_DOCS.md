# Image Processor API — Technical Documentation

> **Audience:** Software developers and AI agents who need to understand,
> debug, extend, or maintain this project.
> **Completeness guarantee:** Reading this document alone is sufficient to
> work on any part of the codebase without looking at external sources.

---

## 1. Project Identity

| Field            | Value                                                       |
|------------------|-------------------------------------------------------------|
| Project name     | image-processor                                             |
| Purpose          | REST API that cleans and isolates product images            |
| Primary use case | E-commerce product photography (e.g., bicycles)             |
| API framework    | FastAPI 0.132+                                              |
| Language         | Python 3.11+                                                |
| Root path        | `D:\Project\image-processor\`                               |
| Entry point      | `python -m app.main` → `http://localhost:8000`              |
| Swagger UI       | `http://localhost:8000/docs`                                |

---

## 2. Goals & Non-Goals

### Goals
- Accept any JPEG / PNG / WEBP image via HTTP POST
- Run a configurable 4-step pipeline: **Denoise → Shadow Removal → Background Removal → Enhancement**
- Return the processed image as PNG / WEBP / JPEG with transparent background
- Preserve fine structural details (bicycle spokes, hair, thin branches)
- Be extensible: each processing step is an independent, replaceable module
- Be deployable as a standalone service (Docker) or embedded in a larger system

### Non-Goals
- Video processing
- Batch processing endpoint (can be added via Celery workers — stubs exist)
- Training or fine-tuning AI models
- Cloud storage integration
- Authentication / rate limiting (add via middleware when needed)

---

## 3. Technology Stack

### Why each technology was chosen

| Layer               | Technology          | Version   | Reason for Choice                                                                 |
|---------------------|---------------------|-----------|-----------------------------------------------------------------------------------|
| API framework       | FastAPI             | ≥0.132    | Async, auto Swagger UI, Pydantic validation, production-ready                     |
| ASGI server         | Uvicorn             | ≥0.41     | Standard pairing with FastAPI, supports multiple workers                          |
| Config management   | pydantic-settings   | ≥2.13     | .env parsing with type validation, zero boilerplate                               |
| Deep learning       | PyTorch             | ≥2.5      | Ecosystem standard; all models require it                                         |
| Image I/O           | Pillow              | ≥12.0     | Universal image format support; pipeline intermediate format                      |
| Computer vision     | OpenCV              | ≥4.13     | Fast CPU ops: denoising fallback, shadow detection, morphology, bilateral filter  |
| Tensor transforms   | torchvision         | ≥0.20     | Resize + Normalize + ToTensor for model input prep                               |
| Model hub           | transformers + HF   | ≥5.2      | BiRefNet loaded via `AutoModelForImageSegmentation`                               |
| BG removal model    | **BiRefNet**        | HF hosted | SOTA for high-detail segmentation (IoU 0.87), handles thin structures             |
| Matting refinement  | **PyMatting**       | ≥1.1.15   | Closed-form alpha matting for smooth edges at boundary                            |
| Denoiser model      | **NAFNet** (opt.)   | manual DL | SOTA denoising PSNR 40.3 dB; falls back to OpenCV if weights absent              |
| Super-resolution    | **super-image**     | ≥0.1.7    | A2N / EDSR models, simple install, no basicsr dependency issues                   |
| SR alternative      | Real-ESRGAN (opt.)  | manual    | Best SR quality; requires `basicsr --no-build-isolation`                          |
| Testing             | pytest              | ≥8.3      | Standard Python test framework                                                    |
| Async test support  | pytest-asyncio      | ≥0.24     | Required for `async def` test functions                                           |
| Containerization    | Docker + Compose    | —         | Reproducible deployment; GPU passthrough via `runtime: nvidia`                   |
| Task queue (opt.)   | Celery + Redis      | stubs     | For async heavy processing; not active by default                                 |

---

## 4. System Architecture

### 4.1 Logical Layers

┌─────────────────────────────────────────────────────┐
│ CLIENT LAYER │
│ (curl / frontend / other service / AI agent) │
└───────────────────┬─────────────────────────────────┘
│ HTTP POST multipart/form-data
▼
┌─────────────────────────────────────────────────────┐
│ API LAYER │
│ FastAPI → router.py → schemas.py │
│ - Validates file type & size │
│ - Parses ProcessingConfig from JSON form field │
│ - Returns StreamingResponse with output image │
└───────────────────┬─────────────────────────────────┘
│ PIL.Image + ProcessingConfig
▼
┌─────────────────────────────────────────────────────┐
│ PIPELINE LAYER │
│ ImagePipeline.execute() │
│ Calls processors in fixed order: │
│ 1 → 2 → 3 → 4 (skips if step disabled in config) │
└──────┬──────────┬─────────────┬──────────┬──────────┘
│ │ │ │
▼ ▼ ▼ ▼
Denoiser Shadow BG Remover Enhancer
(step 1) Remover (step 3) (step 4)
(step 2)
│ │ │ │
└──────────┴─────────────┴──────────┘
│
▼
┌─────────────────────────────────────────────────────┐
│ MODEL MANAGER LAYER │
│ Singleton registry — lazy loads & caches models │
│ Handles device resolution (CUDA / CPU) │
│ Manages GPU VRAM (empty_cache on free) │
└─────────────────────────────────────────────────────┘

### 4.2 Processing Order & Rationale
Input Image (RGB)
│
▼ Step 1: Denoise
│ Removes sensor noise BEFORE shadow/mask detection.
│ Reason: Noisy pixels create false positives in
│ shadow masks and confuse edge detectors.
│
▼ Step 2: Shadow Removal
│ Normalizes illumination BEFORE segmentation.
│ Reason: Shadows near object edges cause BiRefNet
│ to incorrectly include shadow in the foreground mask.
│
▼ Step 3: Background Removal
│ BiRefNet generates binary mask on clean input.
│ PyMatting refines the boundary zone (unknown region).
│ Reason: A clean, shadow-free input maximizes
│ edge quality for fine structures.
│ Output: RGBA (alpha = foreground mask)
│
▼ Step 4: Enhancement (disabled by default)
Super-resolution upscaling on RGBA image.
Alpha channel is enhanced separately then re-attached.
Reason: Upscaling AFTER removal avoids wasting compute
on background pixels that will be discarded.

Output Image (RGBA PNG)

### 4.3 Model Loading Strategy
Models are **lazy-loaded**: a model is not loaded until the first request
that requires it. Once loaded, it stays in memory for the lifetime of the
process (singleton via `ModelManager`).

```
First request arrives
│
▼
ModelManager.is_loaded(name)?
│ │
YES NO
│ │
│ load_model()
│ │
▼ ▼
get(name) ◄─── register(name, model)
│
▼
inference()

```

**Why lazy, not eager?**
Eager loading (at startup) would slow down server startup and load all
models even if only one endpoint is called. Lazy loading means the first
request is slower but subsequent requests use cached models.

---

## 5. File Map

Every file in the project, its purpose, and what it exports/imports.
```
image-processor/
│
├── .env # Environment variables (gitignored)
│ # Parsed by app/config.py via pydantic-settings
│
├── requirements.txt # All Python dependencies with minimum versions
│
├── pytest.ini # pytest configuration: asyncio_mode=auto, testpaths=tests
│
├── Dockerfile # Python 3.11-slim + OpenCV system deps + pip install
│
├── docker-compose.yml # Service: api (port 8000), volume: ./models
│
├── test_main.http # HTTP client test file (PyCharm / IntelliJ format)
│ # Contains test requests for all endpoints
│
├── models/ # Model weight files (NOT committed to git)
│ └── NAFNet-SIDD-width64.pth # Optional: deep denoiser weights
│ # Download: https://github.com/megvii-research/NAFNet
│
├── tests/
│ ├── conftest.py # Shared pytest fixtures: rgb_image, noisy_image,
│ │ # shadowed_image, object_image
│ ├── test_denoiser.py # Tests for DenoiseProcessor._opencv_denoise()
│ ├── test_shadow_remover.py # Tests for ShadowRemovalProcessor._retinex_correction()
│ ├── test_bg_remover.py # Tests for BackgroundRemovalProcessor (mask, trimap, apply)
│ └── test_pipeline.py # Integration tests for ImagePipeline.execute()
│
└── app/
├── init.py # Empty
├── main.py # FastAPI app, lifespan (init/cleanup), CORS, routes
├── config.py # Settings class (pydantic-settings), all env vars
│
├── api/
│ ├── init.py # Empty
│ ├── schemas.py # ProcessingConfig, OutputFormat, EnhanceScale (Pydantic)
│ └── router.py # POST /process, GET /models endpoints
│
├── core/
│ ├── init.py # Empty
│ ├── model_manager.py # Singleton: device resolution, model registry, VRAM mgmt
│ └── pipeline.py # ImagePipeline: orchestrates processor execution order
│
├── processors/
│ ├── init.py # Imports & re-exports all 4 processor classes
│ ├── base.py # BaseProcessor ABC: load_model(), process(), _ensure_model_loaded()
│ ├── denoiser.py # DenoiseProcessor: NAFNet (if weights exist) → OpenCV fallback
│ ├── shadow_remover.py # ShadowRemovalProcessor: LAB + Retinex + Gaussian blend
│ ├── bg_remover.py # BackgroundRemovalProcessor: BiRefNet + PyMatting/bilateral
│ └── enhancer.py # EnhancementProcessor: super-image A2N/EDSR → PIL fallback
│
├── utils/
│ ├── init.py # Empty
│ ├── image_utils.py # validate_image, pil_to_bytes, pil_to_np, resize_long_side
│ └── file_utils.py # ensure_dir, unique_filename, write_temp, delete_file
│
└── workers/
├── init.py # Empty
└── celery_worker.py # Celery app stub (commented out, activate for async tasks)

```

---

## 6. Data Flow

### 6.1 Full Request Lifecycle
```
POST /api/v1/process
multipart/form-data:
file = <image bytes>
config = '{"denoise": true, "remove_background": true, ...}'
│
▼
router.py: _parse_config(config)
→ ProcessingConfig(**json.loads(config))
→ Falls back to ProcessingConfig() defaults if invalid JSON
│
▼
router.py: content = await file.read()
→ size check: len(content) > MAX_FILE_SIZE_MB * 1024^2 → 413
→ Image.open(BytesIO(content)).convert("RGB")
→ validate_image(image, MAX_IMAGE_SIZE)
→ width > 4096 or height > 4096 → 400
→ width < 16 or height < 16 → 400
│
▼
ImagePipeline().execute(image, config)
→ Step 1 (if config.denoise):
denoiser.process(image)
→ _ensure_model_loaded() → load NAFNet or register None
→ if model: _nafnet_denoise(image, model)
→ else: _opencv_denoise(image)
→ returns RGB Image
→ Step 2 (if config.remove_shadow):
shadow_remover.process(image)
→ _retinex_correction(image)
→ cvtColor RGB→LAB
→ detect shadow mask (L channel threshold)
→ adaptive brightness correction in shadow region
→ Gaussian blend at boundary
→ cvtColor LAB→RGB
→ returns RGB Image
→ Step 3 (if config.remove_background):
bg_remover.process(image, use_matting=config.use_matting)
→ _birefnet_mask(image)
→ transform: Resize(1024) + ToTensor + Normalize
→ model(tensor) → list or tensor → [-1].sigmoid()
→ resize mask back to original size
→ returns np.ndarray uint8
→ _refine_mask(image, rough_mask)
→ _build_trimap(mask, border_size=20)
→ erode → sure_fg (255)
→ dilate → expanded
→ outside expanded → bg (0)
→ between eroded and dilated → unknown (128)
→ estimate_alpha_cf(img_np, trimap_f)
→ closed-form alpha matting
→ returns float64 alpha
​
→ _apply_mask(image, final_mask)
→ convert to RGBA, set alpha channel = mask
→ returns RGBA Image
→ Step 4 (if config.enhance):
enhancer.process(image, scale=config.enhance_scale)
→ PIL → tensor [1,C,H,W] / 255.0
→ model(tensor) → squeeze → clamp → PIL
→ if RGBA: enhance RGB only, resize alpha, re-attach
→ returns RGBA Image
→ return (result_RGBA, ["denoise", "shadow_removal", ...])
│
▼
router.py: pil_to_bytes(result, fmt)
→ PNG: compress_level=6
→ WEBP: quality=95, lossless if RGBA
→ JPEG: flatten alpha on white, quality=95
│
▼
StreamingResponse(
BytesIO(output_bytes),
media_type="image/png",
headers={
"X-Processing-Time-Ms": "342.1",
"X-Steps-Applied": "denoise,shadow_removal,background_removal",
"Content-Disposition": 'inline; filename="processed.png"'
}
)
```
---

## 7. Component Deep-Dive

### 7.1 DenoiseProcessor (`app/processors/denoiser.py`)

**Purpose:** Remove image noise while preserving edges and textures.

**Fallback chain:**
NAFNet (if models/NAFNet-SIDD-width64.pth exists AND basicsr installed)
↓ (if not)
OpenCV fastNlMeansDenoisingColored

**NAFNet details:**
- Architecture: Nonlinear Activation Free Network (NAFNet)
- Weights file: `models/NAFNet-SIDD-width64.pth`
- Input/output: float32 tensor [1, 3, H, W] normalized [0,1]
- PSNR: 40.30 dB on SIDD dataset
- Download URL: https://github.com/megvii-research/NAFNet/releases

**OpenCV details:**
- Function: `cv2.fastNlMeansDenoisingColored`
- `h` parameter: maps `DENOISE_STRENGTH` (0.0→1.0) to range (3→15)
- templateWindowSize=7, searchWindowSize=21

**Key config:**
- `DENOISE_STRENGTH` (float 0.0-1.0): only affects OpenCV path

---

### 7.2 ShadowRemovalProcessor (`app/processors/shadow_remover.py`)

**Purpose:** Normalize illumination and remove cast shadows from product images.

**Algorithm (OpenCV Retinex-inspired):**

Step A: RGB → LAB colorspace
Reason: L channel isolates luminance from color.
Shadows appear as low-L regions regardless of object color.

Step B: Shadow mask detection
mean_L = average luminance of entire image
threshold = max(mean_L * 0.65, 30.0)
shadow_mask = pixels where L < threshold
Morphological cleanup (close holes, remove small blobs)
kernel = ellipse 15×15

Step C: Adaptive brightness correction
shadow_mean = mean L in shadow region
lit_mean = mean L in non-shadow region
factor = min(lit_mean / shadow_mean, 1.8) ← capped at 1.8
L_corrected[shadow] = L_original[shadow] * factor

Step D: Gaussian boundary blending
mask_blurred = GaussianBlur(mask, sigmaX=20)
L_final = L_original * (1 - mask_blurred) + L_corrected * mask_blurred
Reason: Avoids hard visible line at shadow boundary.

Step E: LAB → RGB

**When shadow is NOT detected:**
If `shadow_mask.sum() == 0`, the original image is returned unchanged.
This avoids unnecessary processing on already-clean images.

**ShadowFormer (future):**
- `USE_SHADOWFORMER=true` in config will activate Transformer-based removal
- Currently unimplemented — falls back to OpenCV with a warning log

---

### 7.3 BackgroundRemovalProcessor (`app/processors/bg_remover.py`)

**Purpose:** Remove background while preserving fine object details.

**Why BiRefNet over alternatives:**

| Model      | IoU (DIS5K) | Fine Details | Notes                            |
|------------|-------------|--------------|----------------------------------|
| BiRefNet   | 0.87        | Excellent    | Handles spokes, hair, branches   |
| IS-Net     | 0.82        | Good         | BiRefNet predecessor             |
| U2-Net     | 0.39        | Poor         | Used in old rembg                |
| RMBG-2.0   | Very high   | Good         | Based on BiRefNet, non-commercial|

**Stage 1: BiRefNet mask generation**
```python
# Input preparation
transform = Resize(1024) + ToTensor + Normalize(ImageNet stats)
tensor =[2][1]

# Inference
output = model(tensor)  # returns list OR single tensor
pred_tensor = output[-1] if isinstance(output, (list, tuple)) else output
mask = pred_tensor.sigmoid().squeeze()  # [H, W] float[1]

# Resize to original dimensions
mask_uint8 = (mask.numpy() * 255).astype(uint8)
mask_resized = PIL.resize(original_size, LANCZOS)
```

Stage 2: PyMatting boundary refinement

Critical for fine structures like bicycle spokes.

Trimap construction:
  sure_fg  (255) = erode(mask, kernel_20px)
  sure_bg  (0)   = pixels outside dilate(mask, kernel_20px)
  unknown  (128) = the 40px border zone

Closed-form alpha matting (estimate_alpha_cf):
  Input: original RGB image + trimap
  Output: float64 alpha map[1]
  Algorithm solves a sparse linear system to find
  the optimal alpha value in the unknown region,
  guided by the image's local color statistics.
Fallback chain:
```
PyMatting closed-form matting (best edge quality)
    ↓ (if pymatting not installed)
OpenCV bilateral filter (edge-preserving smooth)
    ↓ (if PyMatting raises any exception)
Rough BiRefNet mask (no refinement)
```
Memory note:
BiRefNet uses ~3.5 GB VRAM in FP16 mode.
Set USE_FP16=false for CPU inference (slower but no VRAM requirement).

7.4 EnhancementProcessor (app/processors/enhancer.py)
Purpose: Increase image resolution and enhance detail sharpness.

Note: enhance=false by default. Enable explicitly when needed.

Fallback chain:
```
super-image A2N or EDSR (pip install super-image)
    ↓ (if not installed or fails)
Real-ESRGAN (requires: pip install basicsr --no-build-isolation && pip install realesrgan)
    ↓ (if not installed or fails)
PIL LANCZOS (always available, no AI quality)
super-image inference:
```
```python
# Input: PIL Image (RGB)
img_np = np.array(image).astype(float32) / 255.0
tensor = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0)  # [1,C,H,W]
output = model(tensor)  # [1,C,H*scale,W*scale]
out_pil = Image.fromarray((output.squeeze(0).permute(1,2,0).clamp(0,1).numpy() * 255).astype(uint8))
```
Alpha channel handling (RGBA input):

```python
# Split RGB and alpha
rgb = image.convert("RGB")
alpha = image.split()[2]
# Enhance only RGB
rgb_enhanced = enhancer.process(rgb, scale=scale)
# Upscale alpha to match
alpha_upscaled = alpha.resize(rgb_enhanced.size, LANCZOS)
# Recombine
result = rgb_enhanced.convert("RGBA")
result.putalpha(alpha_upscaled)
8. Configuration Reference
All settings live in app/config.py as a Settings class.
Values are read from .env file or environment variables.
```
| Variable                | Type  | Default               | Description                                           |
| ----------------------- | ----- | --------------------- | ----------------------------------------------------- |
| APP_NAME                | str   | "Image Processor API" | Shown in Swagger UI title                             |
| APP_VERSION             | str   | "1.0.0"               | Shown in /health response                             |
| DEBUG                   | bool  | false                 | Enables uvicorn reload + DEBUG log level              |
| HOST                    | str   | "0.0.0.0"             | Uvicorn bind address                                  |
| PORT                    | int   | 8000                  | Uvicorn bind port                                     |
| WORKERS                 | int   | 1                     | Keep at 1: GPU memory is not shareable                |
| DEVICE                  | str   | "cuda"                | "cuda" or "cpu". Auto-falls back to CPU if no GPU     |
| USE_FP16                | bool  | true                  | Half precision on GPU. Set false for CPU              |
| MAX_IMAGE_SIZE          | int   | 4096                  | Max pixels per side. Over this → HTTP 400             |
| MAX_FILE_SIZE_MB        | int   | 50                    | Max upload size in MB. Over this → HTTP 413           |
| MODELS_DIR              | Path  | "models"              | Directory for manually downloaded model weights       |
| BIREFNET_MODEL          | str   | "zhengpeng7/BiRefNet" | HuggingFace model ID for BiRefNet                     |
| BIREFNET_INPUT_SIZE     | int   | 1024                  | 512 / 1024 / 2048. Higher = better detail + more VRAM |
| BIREFNET_REFINE_MATTING | bool  | true                  | Run PyMatting after BiRefNet mask                     |
| ENHANCEMENT_MODEL       | str   | "a2n"                 | "a2n" or "edsr" for super-image                       |
| REALESRGAN_SCALE        | int   | 2                     | Upscale factor: 2 or 4                                |
| USE_SHADOWFORMER        | bool  | false                 | true = ShadowFormer (not yet impl.), false = OpenCV   |
| DENOISE_STRENGTH        | float | 0.5                   | 0.0-1.0. Only affects OpenCV path (not NAFNet)        |
| CORS_ORIGINS            | list  | ["*"]                 | In .env: must be JSON array string ["*"]              |

Validation rules enforced at startup:

DEVICE must be "cuda" or "cpu"

REALESRGAN_SCALE must be 2 or 4

DENOISE_STRENGTH must be in [0.0, 1.0]

BIREFNET_INPUT_SIZE must be 512, 1024, or 2048

9. API Reference
Base URL: http://localhost:8000
GET /health
Purpose: Liveness check. Returns server status and loaded models.

Response 200:

json
{
  "status": "ok",
  "version": "1.0.0",
  "device": "cuda",
  "models_loaded": ["birefnet", "denoiser"]
}
GET /api/v1/models
Purpose: List which models are currently loaded in memory.

Response 200:

json
{
  "loaded": ["birefnet"],
  "device": "cuda"
}
POST /api/v1/process
Purpose: Run the image processing pipeline.

Request: multipart/form-data

| Field  | Type          | Required | Description                            |
| ------ | ------------- | -------- | -------------------------------------- |
| file   | binary        | Yes      | Image file (JPEG / PNG / WEBP)         |
| config | string (JSON) | No       | ProcessingConfig JSON. Defaults apply. |

ProcessingConfig fields:

| Field             | Type   | Default | Description                         |
| ----------------- | ------ | ------- | ----------------------------------- |
| denoise           | bool   | true    | Apply noise removal                 |
| remove_shadow     | bool   | true    | Apply shadow normalization          |
| remove_background | bool   | true    | Apply BiRefNet + PyMatting          |
| enhance           | bool   | false   | Apply super-resolution (slow)       |
| enhance_scale     | int    | 2       | 2 or 4 — upscale factor             |
| use_matting       | bool   | true    | PyMatting refinement after BiRefNet |
| output_format     | string | "png"   | "png", "webp", or "jpeg"            |


Response 200: Raw image bytes (binary)

Response headers:

text
Content-Type: image/png
X-Processing-Time-Ms: 342.1
X-Steps-Applied: denoise,shadow_removal,background_removal
X-Output-Format: png
Content-Disposition: inline; filename="processed.png"
Error responses:
| Code | Condition                                      |
| ---- | ---------------------------------------------- |
| 400  | Not an image / image too small / cannot decode |
| 413  | File exceeds MAX_FILE_SIZE_MB                  |
| 500  | Any unhandled exception in the pipeline        |

Example (curl):

bash
curl -X POST http://localhost:8000/api/v1/process \
  -F "file=@bicycle.jpg" \
  -F 'config={"denoise":true,"remove_shadow":true,"remove_background":true,"enhance":false}' \
  --output result.png
Example (Python httpx):

python
import httpx

with open("bicycle.jpg", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/api/v1/process",
        files={"file": ("bicycle.jpg", f, "image/jpeg")},
        data={"config": '{"remove_background": true}'},
    )

with open("result.png", "wb") as out:
    out.write(response.content)

print(response.headers["X-Processing-Time-Ms"])
10. Pipeline Execution Logic
python
# app/core/pipeline.py — execute() simplified pseudocode

async def execute(image: RGB_Image, config: ProcessingConfig):
    steps = []
    current = image.convert("RGB")

    if config.denoise:
        current = denoiser.process(current)   # RGB → RGB
        steps.append("denoise")

    if config.remove_shadow:
        current = shadow_remover.process(current)  # RGB → RGB
        steps.append("shadow_removal")

    if config.remove_background:
        current = bg_remover.process(current, use_matting=config.use_matting)
        # RGB → RGBA (background is transparent)
        steps.append("background_removal")

    if config.enhance:
        current = _enhance_preserving_alpha(current, config.enhance_scale)
        # RGBA → RGBA (upscaled)
        steps.append("enhancement")

    return current.convert("RGBA"), steps
Key invariant: Output is always RGBA regardless of which steps ran.
Steps that don't run are simply skipped — no dummy processing.

11. Model Loading & Memory
ModelManager (app/core/model_manager.py)
Singleton class. All methods are @classmethod. Never instantiate directly.

python
# Key API:
ModelManager.get_device()           # → "cuda" or "cpu"
ModelManager.register("name", obj)  # Store model
ModelManager.get("name")            # Retrieve model (None if not loaded)
ModelManager.is_loaded("name")      # Check before loading
ModelManager.free("name")           # Delete + empty_cache()
ModelManager.cleanup()              # Free all models
ModelManager.vram_free_mb()         # Available VRAM in MB (None on CPU)
ModelManager.list_loaded()          # ["birefnet", "denoiser", ...]
Model Registry Keys

| Key              | Processor                  | Value type                          |
| ---------------- | -------------------------- | ----------------------------------- |
| "denoiser"       | DenoiseProcessor           | NAFNet model OR None (CV fallback)  |
| "shadow_remover" | ShadowRemovalProcessor     | str "opencv_retinex"                |
| "birefnet"       | BackgroundRemovalProcessor | BiRefNet model                      |
| "enhancer"       | EnhancementProcessor       | tuple ("super_image", model, scale) |

VRAM Budget (approximate, FP16)

| Model    | VRAM    |
| -------- | ------- |
| BiRefNet | ~3.5 GB |
| NAFNet   | ~0.5 GB |
| A2N x2   | ~0.3 GB |
| Total    | ~4.3 GB |

If VRAM is insufficient, set USE_FP16=false and DEVICE=cpu.

12. Error Handling Patterns
Processor-level errors
Each processor has a fallback chain. If a model fails to load or
inference throws an exception, the processor logs a warning and
moves to the next fallback. Requests never fail due to model issues
unless ALL fallbacks are exhausted.

API-level errors
text
Invalid file type    → HTTPException(400)
File too large       → HTTPException(413)
Cannot decode image  → HTTPException(400)
Pipeline exception   → HTTPException(500) + logger.exception() for full traceback
Logging conventions
logger.info() — model loaded, step started, request completed

logger.debug() — intermediate steps (only visible with DEBUG=true)

logger.warning() — fallback activated (expected, recoverable)

logger.error() — inference failed but fallback used (unexpected)

logger.exception() — unhandled exception with full traceback

13. Testing Strategy
Test types

| File                   | Type        | Scope                                   |
| ---------------------- | ----------- | --------------------------------------- |
| test_denoiser.py       | Unit        | _opencv_denoise() — no GPU/model needed |
| test_shadow_remover.py | Unit        | _retinex_correction() — pure OpenCV     |
| test_bg_remover.py     | Unit        | _apply_mask(), _build_trimap() — pure   |
|                        |             | process() — mocked BiRefNet model       |
| test_pipeline.py       | Integration | execute() — processors mocked           |


Design principle: Tests do NOT load AI models (too slow, requires GPU).
Model-dependent methods are mocked with unittest.mock.patch.
Only pure NumPy/OpenCV methods are tested with real computation.

Run tests
bash
pytest tests/ -v                    # all tests, verbose
pytest tests/test_bg_remover.py -v  # single file
pytest -k "shadow" -v               # filter by name
pytest --tb=short                   # compact tracebacks
Fixtures (defined in tests/conftest.py)

| Fixture        | Size    | Description                                       |
| -------------- | ------- | ------------------------------------------------- |
| rgb_image      | 128×128 | Solid color (120, 150, 180) — no noise, no shadow |
| noisy_image    | 128×128 | Gray base + Gaussian noise (seed=42)              |
| shadowed_image | 200×200 | Upper half bright (200), lower half dark (70)     |
| object_image   | 256×256 | Dark background (40), bright square object (210)  |

14. Common Failure Modes & Fixes
Server startup failures

| Symptom                                      | Cause                           | Fix                                        |
| -------------------------------------------- | ------------------------------- | ------------------------------------------ |
| KeyError: __version__ during pip install     | basicsr setup.py bug            | pip install basicsr --no-build-isolation   |
| ImportError: cannot import 'Settings'        | pydantic-settings not installed | pip install pydantic-settings              |
| ValidationError: DEVICE                      | .env has invalid value          | Set DEVICE=cuda or DEVICE=cpu              |
| pydantic_core.InitErrorDetails: CORS_ORIGINS | Wrong format in .env            | Set CORS_ORIGINS=["*"] (JSON array string) |

Runtime failures

| Symptom                                                                                      | Cause                           | Fix                                                      |
| -------------------------------------------------------------------------------------------- | ------------------------------- | -------------------------------------------------------- |
| BiRefNet download hangs                                                                      | HuggingFace connection issue    | Set HF_ENDPOINT=https://hf-mirror.com (CN mirror)        |
| CUDA out of memory                                                                           | Not enough VRAM                 | Set USE_FP16=true or DEVICE=cpu                          |
| RuntimeError: Input type (torch.HalfTensor) and weight type (torch.FloatTensor) do not match | FP16 mismatch                   | Ensure both model AND input are half()                   |
| BiRefNet returns single tensor, not list                                                     | Different BiRefNet version      | Already handled: isinstance(output, (list, tuple)) check |
| PyMatting LinearSolver error                                                                 | Image has near-empty foreground | Already handled: falls back to bilateral filter          |
| Output is black/blank                                                                        | Alpha channel all zero          | Check rough_mask.max() — BiRefNet confidence too low     |
| Shadow overcorrection                                                                        | Uniformly dark product image    | Lower DENOISE_STRENGTH or set remove_shadow=false        |

Test failures

| Symptom                         | Cause                 | Fix                                            |
| ------------------------------- | --------------------- | ---------------------------------------------- |
| fixture 'noisy_image' not found | Missing conftest.py   | Create tests/conftest.py with fixtures         |
| ScopeMismatch: async            | asyncio_mode not set  | Add pytest.ini with asyncio_mode = auto        |
| assert strong <= mild fails     | Random seed variation | Check numpy.random.default_rng(42) in conftest |


15. Extension Guide
Add a new processor (e.g., WatermarkRemover)
Step 1: Create app/processors/watermark_remover.py

python
from app.processors.base import BaseProcessor
from PIL import Image

class WatermarkRemovalProcessor(BaseProcessor):
    name = "watermark_remover"

    def load_model(self) -> None:
        from app.core.model_manager import ModelManager
        # Load your model here
        ModelManager.register(self.name, your_model)

    def process(self, image: Image.Image, **kwargs) -> Image.Image:
        self._ensure_model_loaded()
        # Your processing logic here
        return processed_image
Step 2: Add to app/processors/__init__.py

python
from app.processors.watermark_remover import WatermarkRemovalProcessor
__all__ = [..., "WatermarkRemovalProcessor"]
Step 3: Add config to app/api/schemas.py

python
class ProcessingConfig(BaseModel):
    ...
    remove_watermark: bool = Field(False, description="Remove watermarks")
Step 4: Add to app/core/pipeline.py

python
def __init__(self):
    ...
    self.watermark_remover = WatermarkRemovalProcessor()

async def execute(self, image, config):
    ...
    if config.remove_watermark:
        current = self.watermark_remover.process(current)
        steps.append("watermark_removal")
Step 5: Add tests to tests/test_watermark_remover.py

Replace BiRefNet with a different segmentation model
Edit app/processors/bg_remover.py

Change load_model() to load your model

Change _birefnet_mask() to call your model's inference

Keep the output contract: np.ndarray of shape (H, W) dtype uint8 range [0, 255]

The rest of the pipeline (PyMatting, _apply_mask) is unchanged

Change the output to always include white background (no transparency)
In app/utils/image_utils.py, pil_to_bytes():

python
# Add before saving:
if fmt_upper in ("PNG", "WEBP"):
    bg = Image.new("RGB", image.size, (255, 255, 255))
    bg.paste(image, mask=image.split())[2]
    image = bg
16. Dependency Graph


```text
app.main
  └── app.api.router
        ├── app.api.schemas          (Pydantic models — no deps)
        ├── app.core.pipeline
        │     ├── app.processors.denoiser
        │     │     ├── app.processors.base
        │     │     ├── app.core.model_manager
        │     │     └── app.config
        │     ├── app.processors.shadow_remover
        │     │     ├── app.processors.base
        │     │     ├── app.core.model_manager
        │     │     └── app.config
        │     ├── app.processors.bg_remover
        │     │     ├── app.processors.base
        │     │     ├── app.core.model_manager
        │     │     └── app.config
        │     └── app.processors.enhancer
        │           ├── app.processors.base
        │           ├── app.core.model_manager
        │           └── app.config
        ├── app.core.model_manager   (singleton — no deps on processors)
        └── app.utils.image_utils    (pure functions — no deps on core)

app.config                           (no internal deps — only pydantic-settings)
app.utils.file_utils                 (no internal deps — only stdlib)
```
Circular import risk: NONE.
model_manager never imports from processors.
processors only import from model_manager and config (inside methods).

Document version: 1.0 | Project version: 1.0.0 | Last updated: 2026-02-23






