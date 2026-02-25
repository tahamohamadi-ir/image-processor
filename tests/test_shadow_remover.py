import numpy as np

from app.processors.shadow_remover import ShadowRemovalProcessor


def test_output_size_unchanged(shadowed_image):
    p = ShadowRemovalProcessor()
    result = p._retinex_correction(shadowed_image)
    assert result.size == shadowed_image.size


def test_output_is_rgb(shadowed_image):
    p = ShadowRemovalProcessor()
    result = p._retinex_correction(shadowed_image)
    assert result.mode == "RGB"


def test_shadow_region_brightened(shadowed_image):
    """The lower (shadowed) half should be brighter after correction."""
    p = ShadowRemovalProcessor()
    result = p._retinex_correction(shadowed_image)
    before = np.array(shadowed_image)[140:].mean()
    after = np.array(result)[140:].mean()
    assert after >= before


def test_no_shadow_returns_original(rgb_image):
    """Uniform image has no shadow — should return virtually unchanged."""
    p = ShadowRemovalProcessor()
    result = p._retinex_correction(rgb_image)
    diff = np.abs(np.array(result).astype(int) - np.array(rgb_image).astype(int))
    assert diff.mean() < 5.0  # negligible change
