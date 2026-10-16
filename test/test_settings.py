"""All supported variants, capacities and algorithm constructor arguments."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from dmiw import algorithms
from dmiw.settings import WatermarkSettings


@pytest.mark.parametrize(
    "name,variant,capacity",
    [
        ("trustmark", "P", 100),
        ("trustmark", "Q", 100),
        ("invismark", "invismark", 100),
        ("invismark", "isp", 100),
        ("videoseal", "pixelseal", 256),
        ("videoseal", "isp_128", 128),
    ],
)
def test_supported_variant_capacity(name, variant, capacity):
    settings = WatermarkSettings(
        _env_file=None,
        algorithm={"algorithm_name": name, "model": variant, "device": "cpu"},
    )
    assert settings.algorithm.algorithm_name == name
    assert settings.algorithm.model == variant
    assert algorithms.settings2capacity(settings.algorithm) == capacity


@pytest.mark.parametrize("name", ["trustmark", "invismark", "videoseal"])
def test_unknown_variant_is_rejected(name):
    with pytest.raises(ValidationError):
        WatermarkSettings(
            _env_file=None,
            algorithm={"algorithm_name": name, "model": "unsupported"},
        )


def test_unknown_algorithm_is_rejected():
    with pytest.raises(ValidationError):
        WatermarkSettings(
            _env_file=None, algorithm={"algorithm_name": "unsupported"}
        )


def test_registry_passes_validated_constructor_options(monkeypatch, settings):
    received = {}

    class ConstructorSpy:
        def __init__(self, **kwargs):
            received.update(kwargs)

    monkeypatch.setitem(
        algorithms.ALGORITHM_REGISTRY, "trustmark", ConstructorSpy
    )
    model = algorithms.get_algorithm(settings.algorithm)
    assert isinstance(model, ConstructorSpy)
    assert "algorithm_name" not in received
    assert received["model"] == "P"
    assert received["device"] == "cpu"
    assert isinstance(received["model_folder"], Path)
    assert received["model_folder"] == settings.algorithm.model_folder


def test_registry_rejects_unknown_algorithm():
    with pytest.raises(ValueError):
        algorithms.get_algorithm(SimpleNamespace(algorithm_name="unsupported"))
