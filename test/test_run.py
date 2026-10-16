"""Replacement for the original test_run.py; CLI tests without model weights."""

import json

import pytest

from dmiw import __main__ as cli


@pytest.mark.parametrize(
    "arguments",
    [
        ("--help",),
        ("embed", "--help"),
        ("extract", "--help"),
        ("embed", "trustmark", "--help"),
        ("embed", "invismark", "--help"),
        ("embed", "videoseal", "--help"),
        ("extract", "trustmark", "--help"),
        ("extract", "invismark", "--help"),
        ("extract", "videoseal", "--help"),
    ],
)
def test_help_without_weights(arguments, invoke_cli):
    result = invoke_cli(*arguments)
    assert result.code == 0
    assert result.stdout.strip()
    assert not result.stderr


@pytest.mark.parametrize(
    "arguments",
    [
        ("unknown-command",),
        ("embed",),
        ("extract",),
        (
            "embed",
            "-i",
            "in.png",
            "-o",
            "out.png",
            "trustmark",
            "--model",
            "bad",
        ),
        ("extract", "-i", "in.png", "videoseal", "--model", "bad"),
        (
            "embed",
            "-i",
            "in.png",
            "-o",
            "out.png",
            "trustmark",
            "--wm_strength",
            "bad",
        ),
        (
            "extract",
            "-i",
            "in.png",
            "trustmark",
            "--interpolation_before",
            "bad",
        ),
    ],
)
def test_bad_arguments_return_error(arguments, invoke_cli):
    result = invoke_cli(*arguments)
    assert result.code == 2
    assert result.stderr.strip()


def test_no_command_prints_help(invoke_cli):
    result = invoke_cli()
    assert result.code == 0
    assert "embed" in result.stdout and "extract" in result.stdout


@pytest.mark.parametrize("command", ["embed", "extract"])
@pytest.mark.parametrize(
    "algorithm,variant",
    [("trustmark", "Q"), ("invismark", "isp"), ("videoseal", "isp_128")],
)
def test_cli_dispatches_validated_settings_and_outputs_json(
    command, algorithm, variant, invoke_cli, monkeypatch, tmp_path
):
    calls = []

    def task(**kwargs):
        calls.append(kwargs)
        return {"value": 0.123456, "count": 100}

    monkeypatch.setattr(cli, "embed_task", task)
    monkeypatch.setattr(cli, "extract_task", task)
    image = tmp_path / "input with spaces.png"
    output = tmp_path / "output with spaces.png"
    model_dir = tmp_path / "model files"
    arguments = [command, "-i", str(image), "-k", "a5"]
    if command == "embed":
        arguments.extend(["-o", str(output)])
    arguments.extend(
        [
            algorithm,
            "--model",
            variant,
            "--device",
            "cpu",
            "--wm_strength",
            "0.75",
            "--model_folder",
            str(model_dir),
        ]
    )
    result = invoke_cli(*arguments)
    assert result.code == 0
    assert not result.stderr
    assert len(calls) == 1
    received = calls[0]
    assert received["image_path"] == str(image)
    assert received["key_hex"] == "a5"
    if command == "embed":
        assert received["output_path"] == str(output)
    settings = received["settings"].algorithm
    assert settings.algorithm_name == algorithm
    assert settings.model == variant
    assert settings.device == "cpu"
    assert settings.wm_strength == 0.75
    assert settings.model_folder == model_dir
    payload = json.loads(result.stdout[result.stdout.index("{") :])
    assert payload == {"value": 0.123, "count": 100}


@pytest.mark.parametrize(
    "arguments,expected", [(("--help",), 0), (("bad-command",), 2)]
)
def test_module_entry_point_from_another_directory(
    arguments, expected, run_cli
):
    result = run_cli(*arguments)
    assert result.returncode == expected, result.stderr


@pytest.mark.parametrize(
    "value,expected",
    [(0.0, 0.0), (1234.0, 1230.0), (0.012345, 0.0123), (-12.345, -12.3)],
)
def test_significant_digits(value, expected):
    assert cli.round_to_n_significant(value, 3) == pytest.approx(expected)


def test_round_metrics_preserves_nested_structure():
    original = {"a": [0.123456, 2], "b": (-12.345, "text"), "c": True}
    rounded = cli.round_metrics(original)
    assert rounded == {"a": [0.123, 2], "b": (-12.3, "text"), "c": True}
    assert original["a"][0] == 0.123456
