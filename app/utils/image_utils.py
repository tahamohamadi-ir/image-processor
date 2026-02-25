import io

import numpy as np
from fastapi import HTTPException
from PIL import Image


def validate_image(image: Image.Image, max_size: int) -> None:
    w, h = image.size
    if w > max_size or h > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large: {w}×{h} px. Maximum is {max_size} px per side.",
        )
    if w < 16 or h < 16:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small: {w}×{h} px. Minimum is 16 px per side.",
        )


def pil_to_bytes(image: Image.Image, fmt: str = "png") -> bytes:
    buf = io.BytesIO()
    fmt_upper = fmt.upper()

    if fmt_upper == "JPEG" and image.mode == "RGBA":
        # Flatten transparent areas onto white background for JPEG output
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg

    save_kwargs: dict = {}
    if fmt_upper == "PNG":
        save_kwargs = {"compress_level": 6}
    elif fmt_upper == "WEBP":
        save_kwargs = {"quality": 95, "lossless": (image.mode == "RGBA")}
    elif fmt_upper == "JPEG":
        save_kwargs = {"quality": 95, "optimize": True}

    image.save(buf, format=fmt_upper, **save_kwargs)
    return buf.getvalue()


def pil_to_np(image: Image.Image, normalize: bool = False) -> np.ndarray:
    arr = np.array(image)
    return arr.astype(np.float32) / 255.0 if normalize else arr


def np_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def resize_long_side(
    image: Image.Image,
    max_size: int,
) -> tuple[Image.Image, float]:
    """Resize so the longest side equals `max_size`. Returns (image, scale_factor)."""
    w, h = image.size
    if max(w, h) <= max_size:
        return image, 1.0
    scale = max_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS), scale
