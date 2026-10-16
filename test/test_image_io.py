"""Actual PNG/JPEG reading and PNG persistence; no model weights."""

import numpy as np
import pytest
import torch
from PIL import Image

from dmiw.tasks import load_image, save_image


def test_saved_png_preserves_8_bit_rgb(tmp_path, rgb_image):
    path = tmp_path / "round_trip.png"
    save_image(rgb_image, str(path))
    decoded = load_image(str(path))
    torch.testing.assert_close(decoded, rgb_image, rtol=0, atol=0)


def test_arbitrary_float_image_png_quantization(tmp_path):
    original = torch.linspace(0, 1, 3 * 16 * 24).reshape(3, 16, 24)
    path = tmp_path / "float.png"
    save_image(original, str(path))
    decoded = load_image(str(path))
    assert decoded.shape == original.shape
    assert torch.max(torch.abs(decoded - original)).item() <= 0.5 / 255 + 1e-7


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_grayscale_and_rgba_are_loaded_as_rgb(tmp_path, mode):
    shape = (16, 24) if mode == "L" else (16, 24, 4)
    image = np.full(shape, 127, dtype=np.uint8)
    if mode == "RGBA":
        image[..., 3] = 255
    path = tmp_path / (mode + ".png")
    Image.fromarray(image).save(path)
    decoded = load_image(str(path))
    assert decoded.shape == (3, 16, 24)
    assert decoded.dtype == torch.float32
    torch.testing.assert_close(decoded, torch.full_like(decoded, 127 / 255))


def test_jpeg_loading(tmp_path):
    path = tmp_path / "input.jpg"
    Image.new("RGB", (24, 16), (30, 100, 200)).save(path, quality=95)
    decoded = load_image(str(path))
    assert decoded.shape == (3, 16, 24)
    assert torch.isfinite(decoded).all()
    assert 0 <= decoded.min() <= decoded.max() <= 1


def test_missing_image_is_rejected(tmp_path):
    with pytest.raises((OSError, RuntimeError)):
        load_image(str(tmp_path / "missing.png"))
