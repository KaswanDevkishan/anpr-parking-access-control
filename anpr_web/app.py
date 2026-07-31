"""Web routes and configuration for the ANPR access-decision prototype."""

import os
import secrets
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from flask import Flask, render_template

from config import REPOSITORY_ROOT, load_config

from .database import VehicleStore
from .mail import DryRunBackend, HTTPAPIBackend, MailService, SMTPBackend
from .processing import WebProcessor
from .routes import web
from .security import csrf_token, protect_post_requests

EIGHT_MEGABYTES = 8 * 1024 * 1024


def _environment_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _positive_int(value, name, *, allow_blank=False):
    if allow_blank and (value is None or str(value).strip() == ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def create_app(test_config=None, processor=None):
    """Create a configured Flask application without loading OCR eagerly."""
    app = Flask(__name__)
    app.config.from_mapping(
        MAX_CONTENT_LENGTH=EIGHT_MEGABYTES,
        SECRET_KEY=os.environ.get("ANPR_SECRET_KEY") or secrets.token_hex(32),
        SQLITE_PATH=os.environ.get(
            "ANPR_SQLITE_PATH", str(REPOSITORY_ROOT / "data" / "anpr_web.sqlite3")
        ),
        SEED_CSV_PATH=str(REPOSITORY_ROOT / "data" / "vehicles.example.csv"),
        UPLOAD_TEMP_DIR=os.environ.get("ANPR_WEB_TEMP_DIR"),
        ADMIN_USERNAME=os.environ.get("ANPR_ADMIN_USERNAME", ""),
        ADMIN_PASSWORD_HASH=os.environ.get("ANPR_ADMIN_PASSWORD_HASH", ""),
        ADMIN_LOGIN_MAX_ATTEMPTS=5,
        ADMIN_LOCKOUT_SECONDS=60,
        ANPR_TIMEZONE=os.environ.get("ANPR_TIMEZONE", "Asia/Tokyo"),
        ANPR_CHECKPOINT_MODE=os.environ.get("ANPR_CHECKPOINT_MODE", "entry")
        .strip()
        .lower(),
        ANPR_APPLICATION_NAME=os.environ.get(
            "ANPR_APPLICATION_NAME", "Example Campus Parking"
        ).strip(),
        ANPR_PARKING_CAPACITY=os.environ.get("ANPR_PARKING_CAPACITY", "").strip(),
        ANPR_EMAIL_ENABLED=_environment_bool("ANPR_EMAIL_ENABLED"),
        ANPR_EMAIL_BACKEND=os.environ.get("ANPR_EMAIL_BACKEND", "smtp").strip().lower(),
        ANPR_EMAIL_API_KEY=os.environ.get("ANPR_EMAIL_API_KEY", ""),
        ANPR_EMAIL_API_URL=os.environ.get("ANPR_EMAIL_API_URL", "").strip(),
        ANPR_SMTP_HOST=os.environ.get("ANPR_SMTP_HOST", "").strip(),
        ANPR_SMTP_PORT=os.environ.get("ANPR_SMTP_PORT", "587").strip(),
        ANPR_SMTP_USERNAME=os.environ.get("ANPR_SMTP_USERNAME", "").strip(),
        ANPR_SMTP_PASSWORD=os.environ.get("ANPR_SMTP_PASSWORD", ""),
        ANPR_SMTP_USE_TLS=_environment_bool("ANPR_SMTP_USE_TLS", True),
        ANPR_EMAIL_FROM=os.environ.get("ANPR_EMAIL_FROM", "").strip(),
        ANPR_EMAIL_FROM_NAME=os.environ.get(
            "ANPR_EMAIL_FROM_NAME", "Example Campus Parking"
        ).strip(),
        ANPR_EMAIL_TIMEOUT_SECONDS=os.environ.get(
            "ANPR_EMAIL_TIMEOUT_SECONDS", "5"
        ).strip(),
        CAMERA_FRAME_MAX_BYTES=2 * 1024 * 1024,
        ANPR_WEB_OCR_BACKEND=os.environ.get("ANPR_WEB_OCR_BACKEND", "easyocr")
        .strip()
        .lower(),
        ANPR_OCR_API_URL=os.environ.get("ANPR_OCR_API_URL", "").strip(),
        ANPR_OCR_API_KEY=os.environ.get("ANPR_OCR_API_KEY", ""),
        ANPR_OCR_TIMEOUT_SECONDS=os.environ.get(
            "ANPR_OCR_TIMEOUT_SECONDS", "5"
        ).strip(),
        CAMERA_SAMPLE_INTERVAL_SECONDS=0.75,
        CAMERA_STABLE_SAMPLES=3,
        CAMERA_SESSION_TIMEOUT_SECONDS=30,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=True,
    )
    if test_config:
        app.config.update(test_config)
    checkpoint_mode = app.config["ANPR_CHECKPOINT_MODE"]
    if checkpoint_mode not in {"entry", "exit", "selectable"}:
        raise ValueError(
            "ANPR_CHECKPOINT_MODE must be 'entry', 'exit', or 'selectable'"
        )
    try:
        app.config["ANPR_ZONEINFO"] = ZoneInfo(app.config["ANPR_TIMEZONE"])
    except ZoneInfoNotFoundError as exc:
        raise ValueError("ANPR_TIMEZONE must be a valid IANA timezone") from exc
    app.config["ANPR_PARKING_CAPACITY"] = _positive_int(
        app.config["ANPR_PARKING_CAPACITY"],
        "ANPR_PARKING_CAPACITY",
        allow_blank=True,
    )
    warning = ""
    try:
        app.config["ANPR_SMTP_PORT"] = _positive_int(
            app.config["ANPR_SMTP_PORT"], "ANPR_SMTP_PORT"
        )
    except ValueError:
        app.config["ANPR_SMTP_PORT"] = 587
        warning = "ANPR_SMTP_PORT must be a positive integer."
    try:
        app.config["ANPR_EMAIL_TIMEOUT_SECONDS"] = _positive_int(
            app.config["ANPR_EMAIL_TIMEOUT_SECONDS"], "ANPR_EMAIL_TIMEOUT_SECONDS"
        )
    except ValueError:
        app.config["ANPR_EMAIL_TIMEOUT_SECONDS"] = 5
        warning = "ANPR_EMAIL_TIMEOUT_SECONDS must be a positive integer."
    try:
        app.config["ANPR_OCR_TIMEOUT_SECONDS"] = _positive_int(
            app.config["ANPR_OCR_TIMEOUT_SECONDS"], "ANPR_OCR_TIMEOUT_SECONDS"
        )
    except ValueError:
        app.config["ANPR_OCR_TIMEOUT_SECONDS"] = 5
    if app.config["ANPR_EMAIL_ENABLED"]:
        backend_name = app.config["ANPR_EMAIL_BACKEND"]
        if warning:
            pass
        elif backend_name not in {"smtp", "http-api", "dry-run"}:
            warning = "Email is enabled but ANPR_EMAIL_BACKEND is invalid."
        elif backend_name == "smtp" and not app.config["ANPR_SMTP_HOST"]:
            warning = "Email is enabled but ANPR_SMTP_HOST is not configured."
        elif backend_name == "http-api" and (
            not app.config["ANPR_EMAIL_API_KEY"]
            or not app.config["ANPR_EMAIL_API_URL"].startswith("https://")
        ):
            warning = "Email is enabled but the HTTPS email API is not configured."
        elif not app.config["ANPR_EMAIL_FROM"]:
            warning = "Email is enabled but ANPR_EMAIL_FROM is not configured."
    app.config["ANPR_EMAIL_CONFIGURATION_WARNING"] = warning
    app.config["ANPR_EMAIL_READY"] = bool(
        app.config["ANPR_EMAIL_ENABLED"] and not warning
    )

    runtime_config = load_config(
        {
            "ANPR_DATABASE_PATH": str(
                REPOSITORY_ROOT / "data" / "vehicles.example.csv"
            ),
            "ANPR_MATCHING_POLICY": os.environ.get("ANPR_MATCHING_POLICY", "exact"),
            "ANPR_MATCH_TOLERANCE": os.environ.get("ANPR_MATCH_TOLERANCE", "1"),
            "ANPR_OCR_CONFIDENCE_THRESHOLD": os.environ.get(
                "ANPR_OCR_CONFIDENCE_THRESHOLD", "0.5"
            ),
            "ANPR_MIN_OCR_LENGTH": os.environ.get("ANPR_MIN_OCR_LENGTH", "3"),
        }
    )
    if runtime_config.matching_policy != "exact":
        raise ValueError("The public web demo requires exact matching")
    if app.config["ANPR_WEB_OCR_BACKEND"] not in {
        "cloud-vision",
        "easyocr",
        "tesseract",
    }:
        raise ValueError(
            "ANPR_WEB_OCR_BACKEND must be 'cloud-vision', 'easyocr', or 'tesseract'"
        )

    if app.testing:
        app.config["SESSION_COOKIE_SECURE"] = False

    store = VehicleStore(app.config["SQLITE_PATH"])
    store.initialize(app.config["SEED_CSV_PATH"])
    app.extensions["anpr_store"] = store
    app.extensions["anpr_runtime_config"] = runtime_config
    app.extensions["anpr_processor"] = processor or WebProcessor(
        runtime_config,
        backend_name=app.config["ANPR_WEB_OCR_BACKEND"],
        backend_options={
            "api_url": app.config["ANPR_OCR_API_URL"],
            "api_key": app.config["ANPR_OCR_API_KEY"],
            "timeout_seconds": app.config["ANPR_OCR_TIMEOUT_SECONDS"],
        },
    )
    app.extensions["anpr_login_attempts"] = {}
    app.extensions["anpr_camera_samples"] = {}
    backend = app.config.get("ANPR_MAIL_BACKEND")
    if backend is None:
        backend_name = app.config["ANPR_EMAIL_BACKEND"]
        if backend_name == "dry-run":
            backend = DryRunBackend()
        elif backend_name == "http-api":
            backend = HTTPAPIBackend(app.config)
        else:
            backend = SMTPBackend(app.config)
    app.extensions["anpr_mail"] = MailService(app.config, backend)

    temp_path = app.config["UPLOAD_TEMP_DIR"]
    if temp_path:
        Path(temp_path).mkdir(parents=True, exist_ok=True)
    else:
        app.config["UPLOAD_TEMP_DIR"] = tempfile.gettempdir()

    app.register_blueprint(web)
    app.before_request(protect_post_requests)
    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.errorhandler(413)
    def upload_too_large(_error):
        return (
            render_template(
                "index.html",
                error="That file is larger than the 8 MB upload limit.",
            ),
            413,
        )

    return app
