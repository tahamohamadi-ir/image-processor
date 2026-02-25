import logging

import cv2
import numpy as np
import torch
from PIL import Image

from app.config import settings
from app.processors.base import BaseProcessor

logger = logging.getLogger(__name__)

_BIREFNET_KEY = "birefnet"


class BackgroundRemovalProcessor(BaseProcessor):
    """
    Two-stage background removal:
      Stage 1: BiRefNet  — high-quality binary alpha mask
      Stage 2: PyMatting — boundary refinement for fine structures
               (bicycle spokes, hair, thin branches)
    """

    name = "bg_remover"

    # ── Model Loading ─────────────────────────────────────────────────
    def load_model(self) -> None:
        from app.core.model_manager import ModelManager
        from transformers import AutoModelForImageSegmentation

        device = ModelManager.get_device()
        logger.info(f"Loading BiRefNet ({settings.BIREFNET_MODEL}) on {device}...")

        model = AutoModelForImageSegmentation.from_pretrained(
            settings.BIREFNET_MODEL,
            trust_remote_code=True,
        )
        model = model.to(device).eval()

        if settings.USE_FP16 and device == "cuda":
            model = model.half()

        ModelManager.register(_BIREFNET_KEY, model)
        logger.info(f"BiRefNet loaded | VRAM free: {ModelManager.vram_free_mb()} MB")

    # ── Main Entry ────────────────────────────────────────────────────
    def process(
        self,
        image: Image.Image,
        use_matting: bool = True,
        **kwargs,
    ) -> Image.Image:
        self._ensure_model_loaded()

        rough_mask = self._birefnet_mask(image)

        if use_matting and settings.BIREFNET_REFINE_MATTING:
            final_mask = self._refine_mask(image, rough_mask)
        else:
            final_mask = rough_mask

        return self._apply_mask(image, final_mask)

    # ── Stage 1: BiRefNet ─────────────────────────────────────────────
    def _birefnet_mask(self, image: Image.Image) -> np.ndarray:
        import torchvision.transforms as T
        from app.core.model_manager import ModelManager

        model = ModelManager.get(_BIREFNET_KEY)
        device = ModelManager.get_device()
        size = settings.BIREFNET_INPUT_SIZE
        orig_size = image.size  # (W, H)

        transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        if settings.USE_FP16 and device == "cuda":
            tensor = tensor.half()

        with torch.no_grad():
            raw_output = model(tensor)

        # ✅ BiRefNet returns a list OR a single tensor depending on version
        if isinstance(raw_output, (list, tuple)):
            pred_tensor = raw_output[-1]   # last = finest detail
        else:
            pred_tensor = raw_output

        # Shape: [1, 1, H, W] → sigmoid → [H, W] float32 in [0, 1]
        pred = pred_tensor.sigmoid().squeeze().cpu().float()

        # ✅ Manual numpy conversion — avoids deprecated ToPILImage on 2D float tensor
        mask_np = (pred.numpy() * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np, mode="L")
        mask_pil = mask_pil.resize(orig_size, Image.LANCZOS)
        return np.array(mask_pil)

    # ── Stage 2: Refinement ───────────────────────────────────────────
    def _refine_mask(
        self,
        image: Image.Image,
        rough_mask: np.ndarray,
    ) -> np.ndarray:
        try:
            from pymatting import estimate_alpha_cf

            img_np = np.array(image.convert("RGB")).astype(np.float64) / 255.0
            trimap = self._build_trimap(rough_mask, border_size=20)
            trimap_f = trimap.astype(np.float64) / 255.0

            alpha = estimate_alpha_cf(img_np, trimap_f)
            return (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)

        except ImportError:
            logger.info("PyMatting not installed — using bilateral filter refinement.")
            return self._bilateral_refine(image, rough_mask)
        except Exception as exc:
            logger.warning(f"PyMatting failed: {exc} — using rough mask.")
            return rough_mask

    def _build_trimap(
        self,
        mask: np.ndarray,
        border_size: int = 20,
    ) -> np.ndarray:
        """
        Trimap zones:
          255 = definite foreground (eroded core)
            0 = definite background (outside dilated mask)
          128 = unknown transition zone (matting operates here)
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (border_size, border_size)
        )
        fg = (mask > 127).astype(np.uint8)
        sure_fg = cv2.erode(fg, kernel)
        expanded = cv2.dilate(fg, kernel)

        trimap = np.full_like(mask, 128, dtype=np.uint8)
        trimap[sure_fg == 1] = 255
        trimap[expanded == 0] = 0
        return trimap

    def _bilateral_refine(
        self,
        image: Image.Image,
        mask: np.ndarray,
    ) -> np.ndarray:
        return cv2.bilateralFilter(mask, d=15, sigmaColor=80, sigmaSpace=80)

    # ── Apply Alpha ───────────────────────────────────────────────────
    def _apply_mask(
        self,
        image: Image.Image,
        mask: np.ndarray,
    ) -> Image.Image:
        img_rgba = np.array(image.convert("RGBA"))
        img_rgba[:, :, 3] = mask
        return Image.fromarray(img_rgba, mode="RGBA")
