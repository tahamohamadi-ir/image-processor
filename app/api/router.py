import io
import json
import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from app.api.schemas import ProcessingConfig
from app.config import settings
from app.core.model_manager import ModelManager
from app.core.pipeline import ImagePipeline
from app.utils.image_utils import pil_to_bytes, validate_image

logger = logging.getLogger(__name__)
router = APIRouter()

MEDIA_TYPES = {
    "png": "image/png",
    "webp": "image/webp",
    "jpeg": "image/jpeg",
}


def _parse_config(raw: str) -> ProcessingConfig:
    try:
        return ProcessingConfig(**json.loads(raw))
    except Exception:
        return ProcessingConfig()


@router.post(
    "/process",
    summary="Process an image",
    description="Runs the full pipeline: denoise → shadow removal → background removal → enhancement.",
    responses={
        200: {"content": {"image/png": {}, "image/webp": {}, "image/jpeg": {}}},
        400: {"description": "Invalid input"},
        413: {"description": "File too large"},
        500: {"description": "Processing error"},
    },
)
async def process_image(
    file: UploadFile = File(..., description="Input image (JPEG / PNG / WEBP)"),
    config: str = Form(default="{}", description="ProcessingConfig JSON string"),
):
    processing_config = _parse_config(config)

    # ── Validate content type ─────────────────────────────────────────
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")

    # ── Read and size-check ───────────────────────────────────────────
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f} MB (limit: {settings.MAX_FILE_SIZE_MB} MB).",
        )

    # ── Decode ────────────────────────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        validate_image(image, settings.MAX_IMAGE_SIZE)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")

    # ── Run pipeline ──────────────────────────────────────────────────
    start = time.perf_counter()
    try:
        result, steps_applied = await ImagePipeline().execute(image, processing_config)
    except Exception as exc:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")

    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        f"Processed '{file.filename}' | {image.size} | "
        f"steps={steps_applied} | {elapsed_ms} ms"
    )

    # ── Encode and return ─────────────────────────────────────────────
    fmt = processing_config.output_format
    output_bytes = pil_to_bytes(result, fmt)

    return StreamingResponse(
        io.BytesIO(output_bytes),
        media_type=MEDIA_TYPES[fmt],
        headers={
            "X-Processing-Time-Ms": str(elapsed_ms),
            "X-Steps-Applied": ",".join(steps_applied),
            "X-Output-Format": fmt,
            "Content-Disposition": f'inline; filename="processed.{fmt}"',
        },
    )


@router.get("/models", summary="List loaded models", tags=["System"])
async def list_models():
    return {
        "loaded": ModelManager.list_loaded(),
        "device": ModelManager.get_device(),
    }
