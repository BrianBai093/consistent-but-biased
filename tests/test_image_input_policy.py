import base64
from io import BytesIO

import pytest
from PIL import Image

from services.image_loader import normalize_image_orientation, resolve_image_url


@pytest.mark.asyncio
async def test_local_image_is_fully_decoded_and_exif_orientation_is_applied(
    tmp_path,
) -> None:
    source = BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (2, 3), "white").save(source, format="JPEG", exif=exif)
    path = tmp_path / "oriented.jpg"
    path.write_bytes(source.getvalue())

    data_url = await resolve_image_url(str(path))
    payload = base64.b64decode(data_url.split(",", 1)[1])

    with Image.open(BytesIO(payload)) as normalized:
        normalized.load()
        assert normalized.size == (3, 2)
        assert normalized.getexif().get(274, 1) == 1


def test_image_without_exif_rotation_is_not_reencoded_or_enhanced() -> None:
    source = BytesIO()
    Image.new("RGB", (4, 5), (12, 34, 56)).save(source, format="PNG")
    original = source.getvalue()

    assert normalize_image_orientation(original, "image/png") == original
