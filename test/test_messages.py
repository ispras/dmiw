"""Payload conversion: leading zeros, fixed capacity and invalid input."""

import pytest
import torch

from dmiw.tasks import hex_str_to_tensor, tensor_to_hex_str


@pytest.mark.parametrize(
    "text,capacity,expected",
    [
        ("0", 4, "0"),
        ("f", 4, "f"),
        ("000f", 16, "000f"),
        ("0xABcd", 16, "abcd"),
        ("0XFF", 16, "00ff"),
        ("abc", 16, "0abc"),
        ("deadbeaf123", 100, "00000000000000deadbeaf123"),
        ("f" * 64, 256, "f" * 64),
    ],
)
def test_hex_round_trip_preserves_payload(text, capacity, expected):
    bits = hex_str_to_tensor(text, capacity)
    assert bits.shape == (capacity,)
    assert bits.dtype == torch.float32
    assert torch.all((bits == 0) | (bits == 1))
    assert tensor_to_hex_str(bits) == expected


def test_known_hex_bit_order():
    bits = hex_str_to_tensor("a5", 8)
    assert bits.tolist() == [1, 0, 1, 0, 0, 1, 0, 1]


def test_oversized_message_uses_documented_prefix_and_warns(capsys):
    bits = hex_str_to_tensor("abcd", 8)
    assert tensor_to_hex_str(bits) == "ab"
    assert capsys.readouterr().out.strip()


@pytest.mark.parametrize("text", ["", "0x", "xyz", "12g4"])
def test_invalid_hex_is_rejected(text):
    with pytest.raises(ValueError):
        hex_str_to_tensor(text, 100)


def test_tensor_batch_dimension_is_flattened():
    tensor = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]])
    assert tensor_to_hex_str(tensor) == "a5"


def test_bit_count_not_divisible_by_four():
    assert tensor_to_hex_str(torch.tensor([1, 0, 1])) == "5"


def test_empty_tensor_becomes_empty_hex():
    assert tensor_to_hex_str(torch.empty(0)) == ""
