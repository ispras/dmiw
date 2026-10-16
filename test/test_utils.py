"""Image tensor contracts and watermark residual scaling."""

import pytest
import torch

from dmiw.utils import (
    denormalize_image,
    normalize_image,
    overlay_difference,
    resize_torch_img,
)


def test_normalization_preserves_image_on_round_trip(rgb_image):
    normalized = normalize_image(rgb_image)
    assert normalized.shape == (1, *rgb_image.shape)
    assert -1 <= normalized.min() <= normalized.max() <= 1
    torch.testing.assert_close(denormalize_image(normalized), rgb_image)


def test_normalization_endpoints():
    image = torch.tensor([[[0.0, 0.5, 1.0]]]).expand(3, 1, 3).clone()
    expected = torch.tensor([-1.0, 0.0, 1.0]).expand(1, 3, 1, 3)
    torch.testing.assert_close(normalize_image(image), expected)


@pytest.mark.parametrize(
    "mode", ["nearest", "bilinear", "bicubic", "area", "nearest-exact"]
)
def test_resize_preserves_channels_and_constant_values(mode):
    image = torch.full((3, 12, 20), 0.375)
    resized = resize_torch_img(image, [7, 9], mode=mode)
    assert resized.shape == (3, 7, 9)
    assert resized.dtype == image.dtype
    torch.testing.assert_close(resized, torch.full((3, 7, 9), 0.375))


def test_resize_equal_dimensions_returns_independent_image(rgb_image):
    resized = resize_torch_img(rgb_image, list(rgb_image.shape[1:]))
    torch.testing.assert_close(resized, rgb_image)
    original = rgb_image.clone()
    resized.zero_()
    torch.testing.assert_close(rgb_image, original)


@pytest.mark.parametrize("factor", [0.0, 0.5, 1.0, 2.0])
def test_overlay_scales_constant_residual_and_clips(rgb_image, factor):
    cover = torch.full((3, 8, 12), 0.4)
    marked = cover + 0.2
    actual = overlay_difference(rgb_image, cover, marked, factor=factor)
    expected = (rgb_image + factor * 0.2).clamp(0, 1)
    assert actual.shape == rgb_image.shape
    torch.testing.assert_close(actual, expected)


def test_overlay_preserves_spatial_residual_pattern():
    original = torch.full((3, 8, 8), 0.5)
    cover = torch.full((3, 2, 2), 0.5)
    residual = torch.tensor([[-0.2, 0.1], [0.3, -0.1]]).expand(3, 2, 2)
    expected = original + residual.repeat_interleave(4, 1).repeat_interleave(
        4, 2
    )
    actual = overlay_difference(
        original, cover, cover + residual, resize_mode="nearest"
    )
    torch.testing.assert_close(actual, expected)
