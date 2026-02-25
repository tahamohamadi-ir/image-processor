import logging

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.processors.base import BaseProcessor

logger = logging.getLogger(__name__)


class ShadowRemovalProcessor(BaseProcessor):
    """
    Shadow removal pipeline:
      1. Convert to LAB (L = luminance, isolates shadow from color)
      2. Detect shadow regions (adaptive threshold on L channel)
      3. Adaptive brightness correction in shadow areas
      4. Gaussian-blended boundary to avoid hard edges
    """

    name = "shadow_remover"

    def load_model(self) -> None:
        from app.core.model_manager import ModelManager

        if settings.USE_SHADOWFORMER:
            logger.warning("ShadowFormer is not yet integrated — using OpenCV fallback.")

        ModelManager.register(self.name, "opencv_retinex")

    def process(self, image: Image.Image, **kwargs) -> Image.Image:
        self._ensure_model_loaded()
        return self._retinex_correction(image)

    # ── Core Algorithm ────────────────────────────────────────────────
    def _retinex_correction(self, image: Image.Image) -> Image.Image:
        img_np = np.array(image).astype(np.uint8)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)

        shadow_mask = self._detect_shadow_mask(l_channel)

        # If no shadow detected, return original unchanged
        if shadow_mask.sum() == 0:
            logger.debug("No shadow regions detected.")
            return image

        l_corrected = self._adaptive_correction(l_channel, shadow_mask)
        l_blended = self._blend_at_boundary(l_channel, l_corrected, shadow_mask)

        lab[:, :, 0] = np.clip(l_blended, 0, 255).astype(np.uint8)
        result_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(result_np)

    # ── Shadow Detection ──────────────────────────────────────────────
    def _detect_shadow_mask(self, l_channel: np.ndarray) -> np.ndarray:
        mean_l = float(l_channel.mean())

        # Pixels more than 35% darker than the mean are considered shadow
        threshold = max(mean_l * 0.65, 30.0)
        raw_mask = (l_channel < threshold).astype(np.uint8) * 255

        # Morphological cleanup — close small holes, remove tiny blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    # ── Brightness Correction ─────────────────────────────────────────
    def _adaptive_correction(
        self,
        l_channel: np.ndarray,
        shadow_mask: np.ndarray,
    ) -> np.ndarray:
        shadow_px = shadow_mask > 0
        lit_px = ~shadow_px

        shadow_mean = float(l_channel[shadow_px].mean())
        lit_mean = (
            float(l_channel[lit_px].mean()) if lit_px.any() else shadow_mean + 40.0
        )

        # Cap factor at 1.8 to prevent over-brightening
        factor = min(lit_mean / (shadow_mean + 1e-6), 1.8)

        l_corrected = l_channel.copy()
        l_corrected[shadow_px] = np.clip(l_channel[shadow_px] * factor, 0, 255)
        return l_corrected

    # ── Boundary Blending ────────────────────────────────────────────
    def _blend_at_boundary(
        self,
        l_original: np.ndarray,
        l_corrected: np.ndarray,
        shadow_mask: np.ndarray,
    ) -> np.ndarray:
        """Gaussian-blend at shadow boundary to avoid visible hard lines."""
        mask_f = shadow_mask.astype(np.float32) / 255.0
        # sigmaX=20 gives ~60px smooth transition zone
        mask_blurred = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=20)
        return l_original * (1.0 - mask_blurred) + l_corrected * mask_blurred
