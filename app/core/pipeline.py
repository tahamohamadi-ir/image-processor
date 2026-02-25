import logging

from PIL import Image

from app.api.schemas import ProcessingConfig
from app.processors.bg_remover import BackgroundRemovalProcessor
from app.processors.denoiser import DenoiseProcessor
from app.processors.enhancer import EnhancementProcessor
from app.processors.shadow_remover import ShadowRemovalProcessor

logger = logging.getLogger(__name__)


class ImagePipeline:
    """
    Fixed processing order (each step benefits from the previous):

        1. Denoise       — clean input → better shadow/mask detection
        2. Shadow Remove — normalize illumination → better segmentation edges
        3. BG Remove     — BiRefNet mask + PyMatting refinement
        4. Enhance       — Real-ESRGAN/A2N on the final clean image
    """

    def __init__(self) -> None:
        self.denoiser = DenoiseProcessor()
        self.shadow_remover = ShadowRemovalProcessor()
        self.bg_remover = BackgroundRemovalProcessor()
        self.enhancer = EnhancementProcessor()

    async def execute(
        self,
        image: Image.Image,
        config: ProcessingConfig,
    ) -> tuple[Image.Image, list[str]]:
        steps: list[str] = []
        current: Image.Image = image.convert("RGB")

        if config.denoise:
            logger.info("[Pipeline] 1/4 Denoising")
            current = self.denoiser.process(current)
            steps.append("denoise")

        if config.remove_shadow:
            logger.info("[Pipeline] 2/4 Shadow removal")
            current = self.shadow_remover.process(current)
            steps.append("shadow_removal")

        if config.remove_background:
            logger.info("[Pipeline] 3/4 Background removal")
            current = self.bg_remover.process(current, use_matting=config.use_matting)
            steps.append("background_removal")
            # current is now RGBA

        if config.enhance:
            logger.info("[Pipeline] 4/4 Enhancement")
            current = self._enhance_preserving_alpha(current, config.enhance_scale)
            steps.append("enhancement")

        # Always output RGBA so the transparent background is preserved
        return current.convert("RGBA"), steps

    # ── Helpers ───────────────────────────────────────────────────────
    def _enhance_preserving_alpha(
        self, image: Image.Image, scale: int
    ) -> Image.Image:
        """Enhance only RGB channels, then re-attach upscaled alpha."""
        if image.mode != "RGBA":
            return self.enhancer.process(image, scale=scale)

        rgb = image.convert("RGB")
        alpha = image.split()[3]

        rgb_enhanced = self.enhancer.process(rgb, scale=scale)

        alpha_upscaled = alpha.resize(rgb_enhanced.size, Image.LANCZOS)
        result = rgb_enhanced.convert("RGBA")
        result.putalpha(alpha_upscaled)
        return result
