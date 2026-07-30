"""Validated runtime configuration and repository-relative paths."""

import os
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent


def _env_int(environ, name, default, minimum=None):
    raw = environ.get(name)
    try:
        value = default if raw in (None, "") else int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _env_float(environ, name, default, minimum=None, maximum=None):
    raw = environ.get(name)
    try:
        value = default if raw in (None, "") else float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _repository_path(value, default):
    path = Path(value or default).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


@dataclass(frozen=True)
class Config:
    camera_index: int
    database_path: Path
    ocr_every_n: int
    ocr_confidence_threshold: float
    matching_policy: str
    match_tolerance: int
    screenshot_directory: Path
    cooldown_seconds: float
    min_ocr_length: int


def load_config(environ=None):
    """Build configuration from an environment mapping using safe defaults."""
    environ = os.environ if environ is None else environ
    policy = environ.get("ANPR_MATCHING_POLICY", "exact").strip().lower()
    if policy not in {"exact", "fuzzy"}:
        raise ValueError("ANPR_MATCHING_POLICY must be 'exact' or 'fuzzy'")

    return Config(
        camera_index=_env_int(environ, "ANPR_CAMERA_INDEX", 0, minimum=0),
        database_path=_repository_path(
            environ.get("ANPR_DATABASE_PATH"), "data/vehicles.example.csv"
        ),
        ocr_every_n=_env_int(environ, "ANPR_OCR_EVERY_N", 30, minimum=1),
        ocr_confidence_threshold=_env_float(
            environ, "ANPR_OCR_CONFIDENCE_THRESHOLD", 0.5, minimum=0, maximum=1
        ),
        matching_policy=policy,
        match_tolerance=_env_int(environ, "ANPR_MATCH_TOLERANCE", 1, minimum=0),
        screenshot_directory=_repository_path(
            environ.get("ANPR_SCREENSHOT_DIR"), "screenshots"
        ),
        cooldown_seconds=_env_float(environ, "ANPR_COOLDOWN_SECONDS", 3.0, minimum=0),
        min_ocr_length=_env_int(environ, "ANPR_MIN_OCR_LENGTH", 3, minimum=1),
    )
