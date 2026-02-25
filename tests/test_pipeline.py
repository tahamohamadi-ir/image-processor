import pytest
from PIL import Image
from unittest.mock import patch

from app.api.schemas import ProcessingConfig
from app.core.pipeline import ImagePipeline


async def _run(image: Image.Image, config: ProcessingConfig):
    return await ImagePipeline().execute(image, config)


@pytest.mark.asyncio
async def test_empty_pipeline_returns_rgba(rgb_image):
    cfg = ProcessingConfig(
        denoise=False,
        remove_shadow=False,
        remove_background=False,
        enhance=False,
    )
    result, steps = await _run(rgb_image, cfg)
    assert result.mode == "RGBA"
    assert steps == []


@pytest.mark.asyncio
async def test_correct_step_labels(rgb_image):
    cfg = ProcessingConfig(
        denoise=True,
        remove_shadow=True,
        remove_background=False,
        enhance=False,
    )
    pipeline = ImagePipeline()
    with (
        patch.object(pipeline.denoiser, "_ensure_model_loaded"),
        patch.object(pipeline.denoiser, "process", return_value=rgb_image),
        patch.object(pipeline.shadow_remover, "_ensure_model_loaded"),
        patch.object(pipeline.shadow_remover, "process", return_value=rgb_image),
    ):
        _, steps = await pipeline.execute(rgb_image, cfg)

    assert steps == ["denoise", "shadow_removal"]


@pytest.mark.asyncio
async def test_bg_removal_produces_rgba(rgb_image):
    cfg = ProcessingConfig(
        denoise=False,
        remove_shadow=False,
        remove_background=True,
        enhance=False,
        use_matting=False,
    )
    pipeline = ImagePipeline()
    rgba_mock = rgb_image.convert("RGBA")

    with (
        patch.object(pipeline.bg_remover, "_ensure_model_loaded"),
        patch.object(pipeline.bg_remover, "process", return_value=rgba_mock),
    ):
        result, steps = await pipeline.execute(rgb_image, cfg)

    assert result.mode == "RGBA"
    assert "background_removal" in steps


@pytest.mark.asyncio
async def test_output_size_preserved(rgb_image):
    """Output dimensions must match input dimensions (no upscaling unless enhance=True)."""
    cfg = ProcessingConfig(
        denoise=True,
        remove_shadow=True,
        remove_background=False,
        enhance=False,
    )
    pipeline = ImagePipeline()
    with (
        patch.object(pipeline.denoiser, "_ensure_model_loaded"),
        patch.object(pipeline.denoiser, "process", return_value=rgb_image),
        patch.object(pipeline.shadow_remover, "_ensure_model_loaded"),
        patch.object(pipeline.shadow_remover, "process", return_value=rgb_image),
    ):
        result, _ = await pipeline.execute(rgb_image, cfg)

    assert result.size == rgb_image.size
