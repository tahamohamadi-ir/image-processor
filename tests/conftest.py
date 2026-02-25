import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def rgb_image() -> Image.Image:
    """Plain solid-color 128×128 RGB image."""
    return Image.new("RGB", (128, 128), color=(120, 150, 180))


@pytest.fixture
def noisy_image() -> Image.Image:
    """128×128 RGB image with Gaussian noise."""
    base = np.full((128, 128, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(42)
    noise = rng.integers(-30, 31, base.shape, dtype=np.int16)
    return Image.fromarray(
        np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    )


@pytest.fixture
def shadowed_image() -> Image.Image:
    """200×200 RGB image with a simulated shadow in the lower half."""
    img = np.full((200, 200, 3), 200, dtype=np.uint8)
    img[120:, :] = 70
    return Image.fromarray(img)


@pytest.fixture
def object_image() -> Image.Image:
    """256×256 RGB image with a bright square object on a dark background."""
    img = np.full((256, 256, 3), 40, dtype=np.uint8)
    img[64:192, 64:192] = 210
    return Image.fromarray(img)
