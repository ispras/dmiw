"""Opt-in inference with the real local checkpoints and PNG round trip."""

import gc
import hashlib
import json
import math

import pytest
import torch

from dmiw.algorithms import get_algorithm, settings2capacity
from dmiw.settings import WatermarkSettings
from dmiw.tasks import load_image, save_image

pytestmark = pytest.mark.integration

CASES = [
    "trustmark-p",
    "trustmark-q",
    "invismark",
    "invismark-isp",
    "pixelseal",
    "videoseal-isp-128",
]


def require_verified_files(case, model_root, explicitly_selected):
    for item in case["files"]:
        path = model_root / item["path"]
        if not path.is_file():
            message = f"Local model file is missing: {path}"
            if explicitly_selected:
                pytest.fail(message)
            pytest.skip(message)
        with path.open("rb") as stream:
            prefix = stream.read(100)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            message = f"Git LFS pointer instead of checkpoint: {path}"
            if explicitly_selected:
                pytest.fail(message)
            pytest.skip(message)
        assert path.stat().st_size == item["size"], f"Wrong file size: {path}"
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == item["sha256"], f"Wrong SHA-256: {path}"


def make_test_image():
    """Deterministic RGB texture; this is a smoke test, not a quality benchmark."""
    axis = torch.linspace(0, 1, 512)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    image = torch.stack(
        [
            0.2 + 0.6 * x,
            0.2 + 0.6 * y,
            0.5 + 0.25 * torch.sin(19 * x) * torch.cos(17 * y),
        ]
    )
    generator = torch.Generator().manual_seed(833)
    noise = torch.randn(image.shape, generator=generator) * 0.04
    return (image + noise).clamp(0, 1)


@pytest.mark.parametrize("case_name", CASES)
def test_model_png_round_trip(
    case_name, request, project_root, tmp_path, record_property
):
    selected = request.config.getoption("model_case")
    if selected and case_name not in selected:
        pytest.skip("model variant not selected")
    maximum_ber = request.config.getoption("max_ber")
    if not math.isfinite(maximum_ber) or not 0 <= maximum_ber < 0.5:
        raise pytest.UsageError("--max-ber must be finite and in [0, 0.5)")
    manifest = json.loads(
        (project_root / "models" / "weights.json").read_text(encoding="utf-8")
    )
    case = manifest["models"][case_name]
    model_root = request.config.getoption("models_root").resolve()
    require_verified_files(case, model_root, bool(selected))
    algorithm = case["algorithm"]
    settings = WatermarkSettings(
        _env_file=None,
        algorithm={
            "algorithm_name": algorithm,
            "model": case["variant"],
            "model_folder": model_root / algorithm,
            "device": "cpu",
        },
    )
    capacity = settings2capacity(settings.algorithm)
    assert capacity == case["capacity_bits"]
    key = torch.randint(
        0, 2, (capacity,), generator=torch.Generator().manual_seed(919)
    ).float()
    assert 0 < key.sum() < capacity
    image = make_test_image()
    model = get_algorithm(settings.algorithm)
    try:
        with torch.inference_mode():
            marked = model.embed(image, key)
        assert marked.shape == image.shape
        assert torch.isfinite(marked).all()
        assert 0 <= marked.min() <= marked.max() <= 1
        output = tmp_path / (case_name + ".png")
        save_image(marked, str(output))
        decoded_image = load_image(str(output))
        with torch.inference_mode():
            extracted = model.extract(decoded_image).flatten()
        assert extracted.numel() == capacity
        assert torch.all((extracted == 0) | (extracted == 1))
        ber = (extracted != key).float().mean().item()
        record_property("model", case_name)
        record_property("ber", ber)
        assert (
            ber <= maximum_ber
        ), f"{case_name}: BER={ber:.4f}, allowed={maximum_ber}"
    finally:
        del model
        gc.collect()
