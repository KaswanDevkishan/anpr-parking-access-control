from pathlib import Path

import pytest

from config import REPOSITORY_ROOT, load_config


def test_default_paths_are_repository_relative(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    config = load_config({})

    assert config.database_path == REPOSITORY_ROOT / "data/vehicles.example.csv"
    assert config.screenshot_directory == REPOSITORY_ROOT / "screenshots"
    assert config.matching_policy == "exact"


def test_relative_environment_paths_are_repository_relative():
    config = load_config(
        {
            "ANPR_DATABASE_PATH": "private/vehicles.csv",
            "ANPR_SCREENSHOT_DIR": "local-captures",
        }
    )

    assert config.database_path == REPOSITORY_ROOT / "private/vehicles.csv"
    assert config.screenshot_directory == REPOSITORY_ROOT / "local-captures"


def test_absolute_environment_path_is_preserved(tmp_path):
    database_path = tmp_path / "vehicles.csv"

    config = load_config({"ANPR_DATABASE_PATH": str(database_path)})

    assert config.database_path == database_path
    assert config.database_path.is_absolute()


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("ANPR_CAMERA_INDEX", "-1"),
        ("ANPR_OCR_EVERY_N", "0"),
        ("ANPR_OCR_CONFIDENCE_THRESHOLD", "1.1"),
        ("ANPR_MATCHING_POLICY", "maybe"),
        ("ANPR_MATCH_TOLERANCE", "-1"),
    ],
)
def test_invalid_configuration_is_rejected(variable, value):
    with pytest.raises(ValueError):
        load_config({variable: value})


def test_path_values_are_path_objects():
    config = load_config({})

    assert isinstance(config.database_path, Path)
    assert isinstance(config.screenshot_directory, Path)
