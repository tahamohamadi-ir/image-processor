from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ──────────────────────────────────────────────
    APP_NAME: str = "Image Processor API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── Server ───────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1  # Keep 1 — GPU memory is not sharable across workers

    # ── Device ───────────────────────────────────────────
    DEVICE: str = "cuda"
    USE_FP16: bool = True

    # ── Limits ───────────────────────────────────────────
    MAX_IMAGE_SIZE: int = 4096   # px per side
    MAX_FILE_SIZE_MB: int = 50

    # ── Paths ────────────────────────────────────────────
    MODELS_DIR: Path = Path("models")

    # ── BiRefNet (Background Removal) ────────────────────
    BIREFNET_MODEL: str = "zhengpeng7/BiRefNet"
    BIREFNET_INPUT_SIZE: int = 1024   # 1024 | 2048 (HR — needs more VRAM)
    BIREFNET_REFINE_MATTING: bool = True

    # ── Enhancement ──────────────────────────────────────
    ENHANCEMENT_MODEL: str = "a2n"    # "a2n" | "edsr"
    REALESRGAN_SCALE: int = 2         # 2 | 4

    # ── Shadow Removal ───────────────────────────────────
    USE_SHADOWFORMER: bool = False    # True = ShadowFormer, False = OpenCV Retinex

    # ── Denoiser ─────────────────────────────────────────
    DENOISE_STRENGTH: float = 0.5     # 0.0 (off) → 1.0 (aggressive)

    # ── CORS ─────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["*"]

    # ── Validators ───────────────────────────────────────
    @field_validator("DEVICE")
    @classmethod
    def validate_device(cls, v: str) -> str:
        if v not in ("cuda", "cpu"):
            raise ValueError("DEVICE must be 'cuda' or 'cpu'")
        return v

    @field_validator("REALESRGAN_SCALE")
    @classmethod
    def validate_scale(cls, v: int) -> int:
        if v not in (2, 4):
            raise ValueError("REALESRGAN_SCALE must be 2 or 4")
        return v

    @field_validator("DENOISE_STRENGTH")
    @classmethod
    def validate_denoise(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("DENOISE_STRENGTH must be in [0.0, 1.0]")
        return v

    @field_validator("BIREFNET_INPUT_SIZE")
    @classmethod
    def validate_birefnet_size(cls, v: int) -> int:
        if v not in (512, 1024, 2048):
            raise ValueError("BIREFNET_INPUT_SIZE must be 512, 1024, or 2048")
        return v


settings = Settings()
