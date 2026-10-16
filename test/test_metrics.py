"""Numerical metric checks; the LPIPS network itself is replaced in unit tests."""

import math

import pytest
import torch

from dmiw import metrics
from dmiw.tasks import calculate_metrics


@pytest.mark.parametrize("errors", [0, 1, 25, 50, 100])
@pytest.mark.parametrize("batch_dimension", [False, True])
def test_ber_counts_bit_errors(errors, batch_dimension):
    original = torch.zeros(100)
    extracted = original.clone()
    extracted[:errors] = 1
    if batch_dimension:
        extracted = extracted.unsqueeze(0)
    result = calculate_metrics(original, extracted)
    assert result["BER"] == pytest.approx(errors / 100)
    assert 0 <= result["P-value"] <= 1


def test_psnr_for_identical_images_is_positive_infinity(rgb_image):
    value = metrics.psnr(rgb_image, rgb_image.clone())
    assert math.isinf(value) and value > 0


def test_psnr_matches_constant_error_formula():
    original = torch.zeros(3, 16, 24)
    distorted = torch.full_like(original, 0.1)
    assert metrics.psnr(original, distorted) == pytest.approx(20.0, abs=1e-5)


def test_ssim_identical_image_is_one(rgb_image):
    assert metrics.ssim(rgb_image, rgb_image.clone()) == pytest.approx(1.0)


def test_ssim_detects_intensity_change_and_is_symmetric():
    first = torch.full((3, 16, 24), 0.2)
    second = torch.full_like(first, 0.8)
    forward = metrics.ssim(first, second)
    assert -1 <= forward < 1
    assert forward == pytest.approx(metrics.ssim(second, first))


def test_lpips_normalizes_and_resizes_inputs_without_gradients(
    monkeypatch, rgb_image
):
    calls = []

    class FakeLPIPS:
        def __init__(self, net, verbose):
            assert net == "alex"
            assert verbose is False

        def to(self, device):
            assert device == "cpu"
            return self

        def __call__(self, first, second):
            assert not torch.is_grad_enabled()
            calls.append((first.clone(), second.clone()))
            return torch.tensor(0.125)

    monkeypatch.setattr(metrics, "LPIPS", FakeLPIPS)
    second = torch.full((3, 8, 12), 0.5)
    result = metrics.lpips(rgb_image, second, device="cpu")
    assert result == pytest.approx(0.125)
    assert len(calls) == 1
    first_input, second_input = calls[0]
    torch.testing.assert_close(first_input, (rgb_image * 2 - 1).unsqueeze(0))
    assert second_input.shape == (1, *rgb_image.shape)
    torch.testing.assert_close(second_input, torch.zeros_like(second_input))
