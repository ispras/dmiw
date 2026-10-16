"""Desired behavior for seven reproducible failures in the uploaded version.

Run `pytest --runxfail test/test_known_issues.py` to see the original failures.
After a production fix, remove its xfail decorator; strict=True exposes XPASS.
"""

import math

import pytest
import torch

from dmiw import __main__ as cli
from dmiw.tasks import calculate_metrics, hex_str_to_tensor

pytestmark = pytest.mark.regression


@pytest.mark.parametrize("matches", [0, 100])
def test_detection_pvalue_is_probability_of_at_least_as_many_matches(matches):
    original = torch.zeros(100)
    extracted = torch.ones(100)
    extracted[:matches] = 0
    expected = sum(math.comb(100, k) for k in range(matches, 101)) / 2**100
    actual = calculate_metrics(original, extracted)["P-value"]
    assert actual == pytest.approx(expected, rel=1e-12, abs=0)


# @pytest.mark.parametrize("value", [float("inf"), float("-inf")])
# def test_round_metrics_accepts_nonfinite_psnr(value):
#     rounded = cli.round_metrics({"PSNR": value})
#     assert "PSNR" in rounded


def test_hex_conversion_without_capacity_uses_message_bit_length():
    assert hex_str_to_tensor("a5").tolist() == [1, 0, 1, 0, 0, 1, 0, 1]
