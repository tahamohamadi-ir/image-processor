import uuid
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def unique_filename(prefix: str = "img", ext: str = "png") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}.{ext}"


def write_temp(
    data: bytes,
    ext: str = "png",
    temp_dir: Path = Path("temp"),
) -> Path:
    ensure_dir(temp_dir)
    path = temp_dir / unique_filename("tmp", ext)
    path.write_bytes(data)
    return path


def delete_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
