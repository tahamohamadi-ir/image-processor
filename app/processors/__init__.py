# app/processors/__init__.py
from app.processors.bg_remover import BackgroundRemovalProcessor
from app.processors.denoiser import DenoiseProcessor
from app.processors.enhancer import EnhancementProcessor
from app.processors.shadow_remover import ShadowRemovalProcessor

__all__ = [
    "DenoiseProcessor",
    "ShadowRemovalProcessor",
    "BackgroundRemovalProcessor",
    "EnhancementProcessor",
]
