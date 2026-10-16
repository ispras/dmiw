"""Shared fixtures for the supplied DMIW source tree and model downloader."""

import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

MODEL_CASES = {
    "trustmark-p": ("trustmark", "P", 100),
    "trustmark-q": ("trustmark", "Q", 100),
    "invismark": ("invismark", "invismark", 100),
    "invismark-isp": ("invismark", "isp", 100),
    "pixelseal": ("videoseal", "pixelseal", 256),
    "videoseal-isp-128": ("videoseal", "isp_128", 128),
}


def pytest_addoption(parser):
    group = parser.getgroup("dmiw models")
    group.addoption(
        "--run-models",
        action="store_true",
        default=False,
        help="run real model smoke tests using already downloaded checkpoints",
    )
    group.addoption(
        "--models-root",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="directory containing trustmark/, invismark/ and videoseal/",
    )
    group.addoption(
        "--model-case",
        action="append",
        choices=sorted(MODEL_CASES),
        default=[],
        help="limit real model tests to this variant; may be repeated",
    )
    group.addoption(
        "--max-ber",
        type=float,
        default=0.10,
        help="maximum BER for the real model smoke test (default 0.10)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("run_models"):
        return
    skip = pytest.mark.skip(reason="real models require --run-models")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session", autouse=True)
def limit_cpu_threads():
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """A missing checkpoint or backbone must never cause a download in tests."""

    def blocked(*args, **kwargs):
        raise AssertionError(
            "Network access is disabled in tests. Download weights beforehand."
        )

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture
def rgb_image():
    import torch

    generator = torch.Generator().manual_seed(417)
    return (
        torch.randint(
            0, 256, (3, 32, 48), dtype=torch.uint8, generator=generator
        ).float()
        / 255
    )


@pytest.fixture
def input_png(tmp_path, rgb_image):
    from dmiw.tasks import save_image

    path = tmp_path / "input.png"
    save_image(rgb_image, str(path))
    return path


@pytest.fixture
def settings(tmp_path):
    from dmiw.settings import WatermarkSettings

    return WatermarkSettings(
        _env_file=None,
        algorithm={
            "algorithm_name": "trustmark",
            "model": "P",
            "model_folder": tmp_path / "models" / "trustmark",
            "device": "cpu",
        },
    )


@pytest.fixture
def invoke_cli(monkeypatch, capsys, tmp_path):
    """Call the current main(), which reads sys.argv and may return None."""
    from dmiw import __main__ as cli

    def invoke(*arguments):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["dmiw", *map(str, arguments)])
        try:
            result = cli.main()
        except SystemExit as exc:
            code = exc.code
        else:
            code = 0 if result is None else result
        captured = capsys.readouterr()
        return SimpleNamespace(
            code=code, stdout=captured.out, stderr=captured.err
        )

    return invoke


@pytest.fixture
def run_cli(tmp_path):
    """Exercise python -m dmiw from outside the repository."""

    def run(*arguments):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(SOURCE_ROOT), environment.get("PYTHONPATH", "")]
        )
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TORCH_HOME"] = str(tmp_path / "torch-cache")
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        for name in list(environment):
            if name.startswith("DMIW_"):
                environment.pop(name)
        return subprocess.run(
            [sys.executable, "-m", "dmiw", *map(str, arguments)],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    return run


@pytest.fixture
def fake_algorithm(monkeypatch):
    """Replace neural inference with a deterministic PNG-safe test codec.

    File I/O, settings, key conversion, PSNR, SSIM and task orchestration remain
    real. This fixture tests the pipeline, not neural watermark quality.
    """
    import torch
    from dmiw import algorithms, tasks

    instances = []
    lpips_calls = []

    class TestCodec:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.embedded_key = None
            instances.append(self)

        def embed(self, image, key):
            self.embedded_key = key.clone()
            pixels = (image * 255).round().to(torch.uint8).contiguous()
            flattened = pixels.view(-1)
            flattened[: key.numel()] = (
                flattened[: key.numel()] & 254
            ) | key.to(torch.uint8)
            return pixels.float() / 255

        def extract(self, image):
            pixels = (image * 255).round().to(torch.uint8).flatten()
            return (pixels[:100] & 1).float()

    def test_lpips(original, marked):
        lpips_calls.append((original.clone(), marked.clone()))
        return 0.012345

    monkeypatch.setitem(algorithms.ALGORITHM_REGISTRY, "trustmark", TestCodec)
    monkeypatch.setattr(tasks, "lpips", test_lpips)
    return SimpleNamespace(instances=instances, lpips_calls=lpips_calls)
