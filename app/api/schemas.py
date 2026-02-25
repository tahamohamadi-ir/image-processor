from enum import Enum

from pydantic import BaseModel, Field


class OutputFormat(str, Enum):
    PNG = "png"
    WEBP = "webp"
    JPEG = "jpeg"


class EnhanceScale(int, Enum):
    X2 = 2
    X4 = 4


class ProcessingConfig(BaseModel):
    denoise: bool = Field(True, description="Remove sensor/JPEG noise (NAFNet / OpenCV)")
    remove_shadow: bool = Field(True, description="Normalize illumination and remove cast shadows")
    remove_background: bool = Field(True, description="BiRefNet + PyMatting alpha matting")
    enhance: bool = Field(False, description="Super-resolution upscaling (slow — disabled by default)")
    enhance_scale: EnhanceScale = Field(EnhanceScale.X2, description="Upscale factor: 2x or 4x")
    use_matting: bool = Field(True, description="PyMatting edge refinement after BiRefNet")
    output_format: OutputFormat = Field(OutputFormat.PNG, description="Output image format")

    model_config = {"use_enum_values": True}


class ProcessingResponse(BaseModel):
    """Returned in headers — not the body (body is the raw image bytes)."""
    processing_time_ms: float
    steps_applied: list[str]
    output_format: str
