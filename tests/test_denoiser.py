import numpy as np

from app.processors.denoiser import DenoiseProcessor


def test_output_size_matches_input(noisy_image):
    p = DenoiseProcessor()
    result = p._opencv_denoise(noisy_image)
    assert result.size == noisy_image.size


def test_output_is_rgb(noisy_image):
    p = DenoiseProcessor()
    result = p._opencv_denoise(noisy_image)
    assert result.mode == "RGB"


def test_noise_is_reduced(noisy_image):
    p = DenoiseProcessor()
    result = p._opencv_denoise(noisy_image)
    assert np.array(result).std() < np.array(noisy_image).std()


def test_strong_denoise_smoother(noisy_image, monkeypatch):
    """Higher DENOISE_STRENGTH → smoother output. Uses monkeypatch for safe cleanup."""
    import app.config as cfg

    p = DenoiseProcessor()

    monkeypatch.setattr(cfg.settings, "DENOISE_STRENGTH", 0.2)
    mild = np.array(p._opencv_denoise(noisy_image)).std()

    monkeypatch.setattr(cfg.settings, "DENOISE_STRENGTH", 0.9)
    strong = np.array(p._opencv_denoise(noisy_image)).std()

    assert strong <= mild
