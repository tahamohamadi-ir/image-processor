import logging
from typing import Any

import torch

from app.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    _models: dict[str, Any] = {}
    _device: str | None = None

    # ── Initialization ────────────────────────────────────────────────
    @classmethod
    async def initialize(cls) -> None:
        cls._device = cls._resolve_device()
        cls._log_gpu_info()

    @classmethod
    def _resolve_device(cls) -> str:
        if settings.DEVICE == "cuda" and torch.cuda.is_available():
            return "cuda"
        if settings.DEVICE == "cuda":
            logger.warning("CUDA requested but not available — falling back to CPU.")
        return "cpu"

    @classmethod
    def _log_gpu_info(cls) -> None:
        if cls._device == "cuda":
            props = torch.cuda.get_device_properties(0)
            total_mb = props.total_memory / 1024**2
            logger.info(f"GPU: {props.name} | VRAM: {total_mb:.0f} MB")
        else:
            logger.info("Running on CPU.")

    # ── Device ────────────────────────────────────────────────────────
    @classmethod
    def get_device(cls) -> str:
        if cls._device is None:
            cls._device = cls._resolve_device()
        return cls._device

    # ── Model registry ────────────────────────────────────────────────
    @classmethod
    def register(cls, name: str, model: Any) -> None:
        cls._models[name] = model
        logger.info(f"Model registered: '{name}'")

    @classmethod
    def get(cls, name: str) -> Any | None:
        return cls._models.get(name)

    @classmethod
    def is_loaded(cls, name: str) -> bool:
        return name in cls._models

    @classmethod
    def list_loaded(cls) -> list[str]:
        return [k for k, v in cls._models.items() if v is not None]

    # ── Memory management ─────────────────────────────────────────────
    @classmethod
    def free(cls, name: str) -> None:
        if name in cls._models:
            del cls._models[name]
            cls._empty_cache()
            logger.info(f"Model freed: '{name}'")

    @classmethod
    def cleanup(cls) -> None:
        cls._models.clear()
        cls._empty_cache()
        logger.info("All models released.")

    @classmethod
    def _empty_cache(cls) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def vram_free_mb(cls) -> float | None:
        if not torch.cuda.is_available():
            return None
        free, _ = torch.cuda.mem_get_info(0)
        return round(free / 1024**2, 1)
