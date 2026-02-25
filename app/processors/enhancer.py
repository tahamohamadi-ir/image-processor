import logging

import numpy as np
from PIL import Image

from app.processors.base import BaseProcessor

logger = logging.getLogger(__name__)

_MODEL_KEY = "enhancer"

_SUPER_IMAGE_MODELS = {
    "a2n": {
        2: "eugenesiow/a2n-bam-div2k-scale2",
        4: "eugenesiow/a2n-bam-div2k-scale4",
    },
    "edsr": {
        2: "eugenesiow/edsr-base",
        4: "eugenesiow/edsr",
    },
}


class EnhancementProcessor(BaseProcessor):
    """
    Super-resolution enhancement.

    Priority:
      1. super-image (A2N / EDSR)  — easy install, great quality
      2. Real-ESRGAN (basicsr)     — best quality, optional
      3. PIL LANCZOS               — always-available fallback
    """

    name = "enhancer"

    def load_model(self) -> None:
        if self._try_load_super_image():
            return
        if self._try_load_realesrgan():
            return
        from app.core.model_manager import ModelManager
        logger.info("Enhancer: using PIL LANCZOS fallback.")
        ModelManager.register(_MODEL_KEY, ("pil", None, 2))

    # ── Loader: super-image ───────────────────────────────────────────
    def _try_load_super_image(self) -> bool:
        from app.core.model_manager import ModelManager
        from app.config import settings

        try:
            from super_image import A2nModel, EdsrModel

            scale = settings.REALESRGAN_SCALE
            model_name = settings.ENHANCEMENT_MODEL.lower()
            model_id = _SUPER_IMAGE_MODELS.get(
                model_name, _SUPER_IMAGE_MODELS["a2n"]
            )[scale]

            ModelClass = A2nModel if model_name == "a2n" else EdsrModel
            model = ModelClass.from_pretrained(model_id, scale=scale)

            device = ModelManager.get_device()
            model = model.to(device).eval()

            if settings.USE_FP16 and device == "cuda":
                model = model.half()

            # ✅ Store only (tag, model, scale) — no ImageLoader
            ModelManager.register(_MODEL_KEY, ("super_image", model, scale))
            logger.info(f"super-image ({model_name.upper()} x{scale}) loaded.")
            return True

        except ImportError:
            logger.info("super-image not installed.")
            return False
        except Exception as exc:
            logger.warning(f"super-image load failed: {exc}")
            return False

    # ── Loader: Real-ESRGAN (optional) ───────────────────────────────
    def _try_load_realesrgan(self) -> bool:
        from app.core.model_manager import ModelManager
        from app.config import settings

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            scale = settings.REALESRGAN_SCALE
            urls = {
                2: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                4: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            }
            device = ModelManager.get_device()

            rrdb = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23,
                num_grow_ch=32, scale=scale,
            )
            upsampler = RealESRGANer(
                scale=scale,
                model_path=urls[scale],
                model=rrdb,
                tile=512,
                tile_pad=10,
                pre_pad=0,
                half=(settings.USE_FP16 and device == "cuda"),
            )
            ModelManager.register(_MODEL_KEY, ("realesrgan", upsampler, scale))
            logger.info(f"Real-ESRGAN x{scale} loaded.")
            return True

        except ImportError:
            logger.info("basicsr/realesrgan not installed.")
            return False
        except Exception as exc:
            logger.warning(f"Real-ESRGAN load failed: {exc}")
            return False

    # ── Main Entry ────────────────────────────────────────────────────
    def process(self, image: Image.Image, scale: int = 2, **kwargs) -> Image.Image:
        self._ensure_model_loaded()
        from app.core.model_manager import ModelManager

        data = ModelManager.get(_MODEL_KEY)
        if not isinstance(data, tuple):
            return self._pil_upscale(image, scale)

        tag = data[0]
        if tag == "super_image":
            return self._run_super_image(image, data)
        if tag == "realesrgan":
            return self._run_realesrgan(image, data[1], data[2])
        return self._pil_upscale(image, scale)

    # ── Inference: super-image ────────────────────────────────────────
    def _run_super_image(self, image: Image.Image, data: tuple) -> Image.Image:
        import torch
        from app.core.model_manager import ModelManager
        from app.config import settings

        _, model, scale = data
        device = ModelManager.get_device()

        try:
            # PIL → tensor [1, C, H, W] in [0.0, 1.0]
            img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
            tensor = (
                torch.from_numpy(img_np)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )
            if settings.USE_FP16 and device == "cuda":
                tensor = tensor.half()

            with torch.no_grad():
                output = model(tensor)  # [1, C, H_out, W_out]

            # ✅ tensor → numpy → PIL (no ImageLoader dependency)
            out_np = (
                output.squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .float()
                .clamp(0.0, 1.0)
                .numpy()
            )
            return Image.fromarray((out_np * 255).astype(np.uint8))

        except Exception as exc:
            logger.error(f"super-image inference failed: {exc}. Falling back.")
            return self._pil_upscale(image, scale)

    # ── Inference: Real-ESRGAN ────────────────────────────────────────
    def _run_realesrgan(self, image: Image.Image, upsampler, scale: int) -> Image.Image:
        try:
            img_bgr = np.array(image)[:, :, ::-1]
            output_bgr, _ = upsampler.enhance(img_bgr, outscale=scale)
            return Image.fromarray(output_bgr[:, :, ::-1])
        except Exception as exc:
            logger.error(f"Real-ESRGAN inference failed: {exc}. Falling back.")
            return self._pil_upscale(image, scale)

    # ── Fallback: PIL ─────────────────────────────────────────────────
    def _pil_upscale(self, image: Image.Image, scale: int) -> Image.Image:
        w, h = image.size
        return image.resize((w * scale, h * scale), Image.LANCZOS)
