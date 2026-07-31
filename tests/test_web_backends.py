import json
import subprocess
import sys
import threading
from email.message import EmailMessage
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from anpr_web import create_app
from anpr_web.mail import HTTPAPIBackend, SMTPBackend
from anpr_web.ocr import (
    EasyOCRBackend,
    OCRTimeout,
    TesseractOCRBackend,
    _run_tesseract,
)
from anpr_web.processing import ScannerBusy, WebProcessor


def test_tesseract_selection_does_not_import_heavy_ocr_modules(tmp_path):
    sys.modules.pop("easyocr", None)
    sys.modules.pop("torch", None)
    app = create_app(
        {
            "TESTING": True,
            "SQLITE_PATH": str(tmp_path / "light.sqlite3"),
            "ANPR_WEB_OCR_BACKEND": "tesseract",
        }
    )
    assert "easyocr" not in sys.modules
    assert "torch" not in sys.modules
    assert app.extensions["anpr_processor"]._backend is None


def test_health_and_normal_pages_do_not_initialize_ocr(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "SQLITE_PATH": str(tmp_path / "pages.sqlite3"),
            "ANPR_WEB_OCR_BACKEND": "tesseract",
        }
    )
    processor = app.extensions["anpr_processor"]
    client = app.test_client()
    assert client.get("/health").status_code == 200
    for path in ("/", "/registered-vehicles", "/access-history", "/admin/login"):
        assert client.get(path).status_code == 200
    assert processor._backend is None


def test_tesseract_extracts_longest_mocked_digit_run(monkeypatch):
    completed = SimpleNamespace(returncode=0, stdout=b"12 3456\n")
    monkeypatch.setattr(
        "anpr_web.ocr.subprocess.run", lambda *args, **kwargs: completed
    )
    image = np.zeros((40, 120, 3), dtype=np.uint8)
    assert TesseractOCRBackend().read(image) == "3456"


def test_tesseract_never_exceeds_three_calls(monkeypatch):
    image = np.zeros((40, 120, 3), dtype=np.uint8)
    calls = []
    monkeypatch.setattr(
        "anpr_web.ocr._run_tesseract",
        lambda candidate, psm, timeout: calls.append((psm, timeout)) or "",
    )

    assert TesseractOCRBackend().read(image) == ""
    assert len(calls) == 3
    assert [psm for psm, _timeout in calls] == [7, 7, 13]
    assert all(0 < timeout <= 1.2 for _psm, timeout in calls)


def test_tesseract_early_success_stops_later_calls(monkeypatch):
    calls = []

    def fake_run(_candidate, _psm, timeout):
        calls.append(timeout)
        return "1238"

    monkeypatch.setattr("anpr_web.ocr._run_tesseract", fake_run)
    image = np.zeros((40, 120, 3), dtype=np.uint8)
    assert TesseractOCRBackend().read(image) == "1238"
    assert len(calls) == 1


def test_total_ocr_deadline_stops_before_another_call(monkeypatch):
    now = [0.0]
    calls = []

    def fake_run(_candidate, _psm, timeout):
        calls.append(timeout)
        now[0] += 2.0
        return ""

    monkeypatch.setattr("anpr_web.ocr._run_tesseract", fake_run)
    backend = TesseractOCRBackend(clock=lambda: now[0])
    image = np.zeros((40, 120, 3), dtype=np.uint8)
    with pytest.raises(OCRTimeout):
        backend.read(image)
    assert len(calls) == 2


def test_subprocess_timeout_is_converted_to_ocr_timeout(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("tesseract", 1.2)

    monkeypatch.setattr("anpr_web.ocr.subprocess.run", timeout)
    with pytest.raises(OCRTimeout):
        _run_tesseract(np.zeros((10, 10), dtype=np.uint8), 7, 1.2)


def test_processor_timeout_returns_normal_result(tmp_path):
    processor, image_path = _blocking_processor(
        tmp_path, lambda *_args, **_kwargs: (_ for _ in ()).throw(OCRTimeout())
    )
    result = processor.process(image_path, {})
    assert result.status == "NOT ALLOWED"
    assert (
        result.reason
        == "OCR timed out. Please try a clearer or more tightly cropped image."
    )


def _blocking_processor(tmp_path, ocr):
    image_path = tmp_path / "scan.png"
    cv2.imwrite(str(image_path), np.zeros((20, 60, 3), dtype=np.uint8))
    runtime = SimpleNamespace(
        ocr_confidence_threshold=0.5,
        min_ocr_length=3,
        matching_policy="exact",
        match_tolerance=0,
    )
    processor = WebProcessor(
        runtime,
        detector=lambda frame: (frame, (0, 0, 60, 20), "test"),
        ocr=ocr,
    )
    return processor, image_path


def test_simultaneous_scan_returns_scanner_busy(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def slow_ocr(*_args, **_kwargs):
        started.set()
        release.wait(2)
        return ""

    processor, image_path = _blocking_processor(tmp_path, slow_ocr)
    first = threading.Thread(target=processor.process, args=(image_path, {}))
    first.start()
    assert started.wait(1)
    try:
        with pytest.raises(ScannerBusy, match="Scanner busy, try again shortly"):
            processor.process(image_path, {})
    finally:
        release.set()
        first.join(2)


def test_health_responds_while_ocr_is_slow(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def slow_ocr(*_args, **_kwargs):
        started.set()
        release.wait(2)
        return ""

    processor, image_path = _blocking_processor(tmp_path, slow_ocr)
    app = create_app(
        {
            "TESTING": True,
            "SQLITE_PATH": str(tmp_path / "health.sqlite3"),
        },
        processor=processor,
    )
    scan = threading.Thread(target=processor.process, args=(image_path, {}))
    scan.start()
    assert started.wait(1)
    try:
        response = app.test_client().get("/health")
        assert response.status_code == 200
        assert response.json["status"] == "ok"
    finally:
        release.set()
        scan.join(2)


def test_render_memory_safe_configuration_is_unchanged():
    render_config = (Path(__file__).resolve().parents[1] / "render.yaml").read_text(
        encoding="utf-8"
    )
    for setting in (
        "gunicorn --workers 1 --threads 2",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "MALLOC_ARENA_MAX",
    ):
        assert setting in render_config


def test_exact_authorization_occurs_only_after_ocr_returns_registered_number(
    tmp_path,
):
    image_path = tmp_path / "plate.png"
    cv2.imwrite(str(image_path), np.zeros((20, 60, 3), dtype=np.uint8))
    runtime = SimpleNamespace(
        ocr_confidence_threshold=0.5,
        min_ocr_length=3,
        matching_policy="exact",
        match_tolerance=0,
    )

    def detector(frame):
        return frame, (0, 0, 60, 20), "Close-up fallback"

    database = {"1238": {"name": "Fictional", "id": "X", "type": "test"}}

    empty = WebProcessor(runtime, detector=detector, ocr=lambda *_args, **_kw: "")
    assert empty.process(image_path, database).status == "NOT ALLOWED"
    recognized = WebProcessor(
        runtime, detector=detector, ocr=lambda *_args, **_kw: "1238"
    )
    assert recognized.process(image_path, database).status == "ALLOWED"
    unknown = WebProcessor(runtime, detector=detector, ocr=lambda *_args, **_kw: "9999")
    assert unknown.process(image_path, database).status == "NOT ALLOWED"


def test_easyocr_adapter_remains_available_lazily(monkeypatch):
    reader = object()
    engine = SimpleNamespace(
        load_ocr=lambda: reader,
        read_plate=lambda _image, supplied, confidence_threshold: (
            "1001" if supplied is reader and confidence_threshold == 0.5 else ""
        ),
    )
    monkeypatch.setattr("anpr_web.ocr.importlib.import_module", lambda name: engine)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert EasyOCRBackend().read(image) == "1001"


class FakeHTTPResponse:
    status = 202

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def sample_message():
    message = EmailMessage()
    message["From"] = "Demo <sender@example.test>"
    message["To"] = "owner@example.test"
    message["Subject"] = "Visit completed"
    message.set_content("Text summary")
    message.add_alternative("<p>HTML summary</p>", subtype="html")
    return message


def test_http_email_backend_posts_provider_neutral_json_over_https():
    captured = {}

    def opener(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeHTTPResponse()

    backend = HTTPAPIBackend(
        {
            "ANPR_EMAIL_API_URL": "https://mail.example.test/send",
            "ANPR_EMAIL_API_KEY": "test-api-key",
            "ANPR_EMAIL_TIMEOUT_SECONDS": 5,
        },
        opener=opener,
    )
    backend.send(sample_message())
    payload = json.loads(captured["request"].data)
    assert captured["request"].full_url.startswith("https://")
    assert captured["request"].headers["Authorization"] == "Bearer test-api-key"
    assert payload["to"] == "owner@example.test"
    assert payload["text"].strip() == "Text summary"
    assert "HTML summary" in payload["html"]


def test_smtp_backend_remains_available(monkeypatch):
    sent = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            sent.append((host, port, timeout))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def starttls(self):
            sent.append("tls")

        def login(self, username, password):
            sent.append((username, password))

        def send_message(self, message):
            sent.append(message)

    monkeypatch.setattr("anpr_web.mail.smtplib.SMTP", FakeSMTP)
    SMTPBackend(
        {
            "ANPR_SMTP_HOST": "smtp.example.test",
            "ANPR_SMTP_PORT": 587,
            "ANPR_EMAIL_TIMEOUT_SECONDS": 5,
            "ANPR_SMTP_USE_TLS": True,
            "ANPR_SMTP_USERNAME": "local-user",
            "ANPR_SMTP_PASSWORD": "local-password",
        }
    ).send(sample_message())
    assert sent[0] == ("smtp.example.test", 587, 5)
    assert "tls" in sent


def test_missing_http_api_configuration_disables_email_safely(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "SQLITE_PATH": str(tmp_path / "missing-api.sqlite3"),
            "ANPR_EMAIL_ENABLED": True,
            "ANPR_EMAIL_BACKEND": "http-api",
            "ANPR_EMAIL_API_KEY": "",
            "ANPR_EMAIL_API_URL": "",
            "ANPR_EMAIL_FROM": "sender@example.test",
        }
    )
    assert not app.config["ANPR_EMAIL_READY"]
    assert "HTTPS email API" in app.config["ANPR_EMAIL_CONFIGURATION_WARNING"]
