import numpy as np
import pytest
from PIL import Image
from unittest.mock import patch

from app.processors.bg_remover import BackgroundRemovalProcessor


@pytest.fixture
def processor():
    return BackgroundRemovalProcessor()


def test_apply_full_mask_keeps_all_pixels(processor, object_image):
    mask = np.full((256, 256), 255, dtype=np.uint8)
    result = processor._apply_mask(object_image, mask)
    assert result.mode == "RGBA"
    assert np.array(result)[:, :, 3].min() == 255


def test_apply_empty_mask_all_transparent(processor, object_image):
    mask = np.zeros((256, 256), dtype=np.uint8)
    result = processor._apply_mask(object_image, mask)
    assert np.array(result)[:, :, 3].max() == 0


def test_apply_partial_mask(processor, object_image):
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[64:192, 64:192] = 255
    result = processor._apply_mask(object_image, mask)
    arr = np.array(result)
    assert arr[128, 128, 3] == 255   # inside object → opaque
    assert arr[10, 10, 3] == 0      # outside object → transparent


def test_trimap_has_exactly_three_values(processor):
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255
    trimap = processor._build_trimap(mask, border_size=8)
    unique = set(np.unique(trimap).tolist())
    assert unique.issubset({0, 128, 255})


def test_trimap_has_all_three_regions(processor):
    """Large enough mask should produce all three trimap zones."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[40:160, 40:160] = 255
    trimap = processor._build_trimap(mask, border_size=15)
    assert 0 in trimap
    assert 128 in trimap
    assert 255 in trimap


def test_process_without_matting_returns_rgba(processor, object_image):
    full_mask = np.full((256, 256), 255, dtype=np.uint8)
    with (
        patch.object(processor, "_ensure_model_loaded"),
        patch.object(processor, "_birefnet_mask", return_value=full_mask),
    ):
        result = processor.process(object_image, use_matting=False)
    assert result.mode == "RGBA"
    assert result.size == object_image.size
