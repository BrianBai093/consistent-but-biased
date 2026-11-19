"""Safe conversion of supported local images to Responses API data URLs."""

import asyncio
import base64
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageOps, UnidentifiedImageError

from exceptions import ModelInvocationError


SUPPORTED_IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


async def resolve_image_url(path_or_url: str) -> str:
    parsed = urlparse(path_or_url)
    if parsed.scheme in {"http", "https", "data"}:
        return path_or_url

    path = Path(path_or_url)
    media_type = SUPPORTED_IMAGE_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise ModelInvocationError("Only PNG and JPEG image inputs are supported.")
    if not path.is_file():
        raise ModelInvocationError("A configured image file could not be read.")
    content = await asyncio.to_thread(path.read_bytes)
    try:
        content = await asyncio.to_thread(
            normalize_image_orientation,
            content,
            media_type,
        )
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ModelInvocationError("A configured image file could not be decoded.") from exc
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def validate_image_bytes(content: bytes) -> None:
    """Fully decode an upload so a valid magic prefix cannot hide a corrupt image."""

    with Image.open(BytesIO(content)) as image:
        image.load()


def normalize_image_orientation(content: bytes, media_type: str) -> bytes:
    """Apply EXIF orientation in memory without clinical-image enhancement."""

    with Image.open(BytesIO(content)) as image:
        image.load()
        orientation = image.getexif().get(274, 1)
        if orientation == 1:
            return content
        normalized = ImageOps.exif_transpose(image)
        destination = BytesIO()
        if media_type == "image/jpeg":
            if normalized.mode not in {"L", "RGB", "CMYK"}:
                normalized = normalized.convert("RGB")
            normalized.save(destination, format="JPEG", quality=95, subsampling=0)
        elif media_type == "image/png":
            normalized.save(destination, format="PNG")
        else:
            raise ValueError("Unsupported image media type")
        return destination.getvalue()
