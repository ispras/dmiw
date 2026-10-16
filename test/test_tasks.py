"""Task pipeline checks with real image persistence and a test inference codec."""

import math

import pytest
import torch

from dmiw import tasks


def test_embed_saves_image_and_reports_metrics(
    tmp_path, input_png, settings, fake_algorithm
):
    output = tmp_path / "marked.png"
    result = tasks.embed_task(str(input_png), "a5", str(output), settings)
    assert output.is_file()
    assert result["Внедренное сообщение"] == "0" * 23 + "a5"
    assert 0 < result["SSIM"] <= 1
    assert math.isfinite(result["PSNR"]) and result["PSNR"] > 40
    assert result["LPIPS"] == pytest.approx(0.012345)
    assert result["Время инициализации"] >= 0
    assert result["Время внедрения"] >= 0
    assert len(fake_algorithm.instances) == 1
    assert len(fake_algorithm.lpips_calls) == 1
    embedded = fake_algorithm.instances[0].embedded_key
    assert embedded.shape == (100,)
    assert fake_algorithm.instances[0].options["device"] == "cpu"
    assert tasks.load_image(str(output)).shape == (3, 32, 48)


def test_extract_reads_saved_image_and_recovers_message(
    tmp_path, input_png, settings, fake_algorithm
):
    output = tmp_path / "marked.png"
    key = "deadbeaf123"
    tasks.embed_task(str(input_png), key, str(output), settings)
    result = tasks.extract_task(str(output), key, settings)
    expected = "00000000000000deadbeaf123"
    assert result["Сообщение для сравнения"] == expected
    extracted = next(
        value
        for name, value in result.items()
        if name.strip() == "Извлеченное сообщение"
    )
    assert extracted == expected
    assert result["BER"] == 0
    assert result["P-value"] < 1e-20
    assert result["Время инициализации"] >= 0
    assert result["Время извлечения"] >= 0
    assert len(fake_algorithm.instances) == 2
    assert len(fake_algorithm.lpips_calls) == 1


def test_extract_comparison_key_changes_ber(
    tmp_path, input_png, settings, fake_algorithm
):
    output = tmp_path / "marked.png"
    tasks.embed_task(str(input_png), "f" * 25, str(output), settings)
    result = tasks.extract_task(str(output), "0" * 25, settings)
    assert result["BER"] == 1


def test_invalid_key_never_initializes_model(
    tmp_path, input_png, settings, fake_algorithm
):
    with pytest.raises(ValueError):
        tasks.embed_task(
            str(input_png),
            "invalid-key",
            str(tmp_path / "marked.png"),
            settings,
        )
    assert fake_algorithm.instances == []
    assert not (tmp_path / "marked.png").exists()


def test_saved_marked_image_is_actually_used_for_extraction(
    tmp_path, input_png, settings, fake_algorithm
):
    output = tmp_path / "marked.png"
    key = "f" * 25
    tasks.embed_task(str(input_png), key, str(output), settings)
    # Overwrite the file after embedding: extraction must see its new contents.
    tasks.save_image(torch.zeros(3, 32, 48), str(output))
    result = tasks.extract_task(str(output), key, settings)
    assert result["BER"] == 1
