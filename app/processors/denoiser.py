import logging

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.processors.base import BaseProcessor

logger = logging.getLogger(__name__)

_NAFNET_KEY = "denoiser"


class DenoiseProcessor(BaseProcessor):
    """
    Priority:
      1. NAFNet (SOTA denoiser) — requires manual weight download
      2. OpenCV fastNlMeansDenoisingColored — zero-dependency fallback
    """

    name = _NAFNET_KEY

    def load_model(self) -> None:
        from app.core.model_manager import ModelManager

        if self._try_load_nafnet():
            return
        logger.info("Denoiser: using OpenCV fallback.")
        ModelManager.register(self.name, None)

    def _try_load_nafnet(self) -> bool:
        from app.core.model_manager import ModelManager

        try:
            import torch
            from basicsr.archs.nafnet_arch import NAFNet

            weights_path = settings.MODELS_DIR / "NAFNet-SIDD-width64.pth"
            if not weights_path.exists():
                logger.info(
                    f"NAFNet weights not found at '{weights_path}'. "
                    "Skipping. Download from: https://github.com/megvii-research/NAFNet"
                )
                return False

            device = ModelManager.get_device()
            model = NAFNet(
                img_channel=3,
                width=64,
                middle_blk_num=12,
                enc_blks=[2, 2, 4, 8],
                dec_blks=[2, 2, 2, 2],
            )
            state = torch.load(weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state["params"], strict=False)
            model = model.to(device).eval()
            if settings.USE_FP16 and device == "cuda":
                model = model.half()

            ModelManager.register(self.name, model)
            logger.info("NAFNet loaded — deep denoising active.")
            return True

        except (ImportError, Exception) as e:
            logger.info(f"NAFNet unavailable: {e}")
            return False

    # ── Main entry ────────────────────────────────────────────────────
    def process(self, image: Image.Image, **kwargs) -> Image.Image:
        self._ensure_model_loaded()
        from app.core.model_manager import ModelManager

        model = ModelManager.get(self.name)
        if model is not None:
            return self._nafnet_denoise(image, model)
        return self._opencv_denoise(image)

    # ── NAFNet ────────────────────────────────────────────────────────
    def _nafnet_denoise(self, image: Image.Image, model) -> Image.Image:
        import torch
        from app.core.model_manager import ModelManager

        device = ModelManager.get_device()
        img_np = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        if settings.USE_FP16 and device == "cuda":
            tensor = tensor.half()

        with torch.no_grad():
            out = model(tensor)

        out_np = out.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        out_np = np.clip(out_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(out_np)

    # ── OpenCV ────────────────────────────────────────────────────────
    def _opencv_denoise(self, image: Image.Image) -> Image.Image:
        # h: filter strength — maps [0.0, 1.0] → [3, 15]
        h = max(3, int(settings.DENOISE_STRENGTH * 15))
        img_np = np.array(image)
        denoised = cv2.fastNlMeansDenoisingColored(
            img_np,
            None,
            h=h,
            hColor=h,
            templateWindowSize=7,
            searchWindowSize=21,
        )
        return Image.fromarray(denoised)
