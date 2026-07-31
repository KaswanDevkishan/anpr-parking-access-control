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
    HTTPSOCRBackend,
    OCRSpaceProvider,
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


def _ocr_space_response(text="1238", **overrides):
    payload = {
        "ParsedResults": [{"ParsedText": text}],
        "OCRExitCode": 1,
        "IsErroredOnProcessing": False,
        "ErrorMessage": None,
    }
    payload.update(overrides)
    return payload


def _ocr_space_backend(response=None, error=None, api_key="test-secret"):
    def transport(_request, _timeout):
        if error:
            raise error
        return response

    return HTTPSOCRBackend(
        OCRSpaceProvider("https://api.ocr.space/parse/image", api_key),
        transport=transport,
    )


def test_ocr_space_success_returns_1238():
    backend = _ocr_space_backend(_ocr_space_response())
    assert backend.read(np.zeros((20, 60, 3), dtype=np.uint8)) == "1238"


def test_ocr_space_request_is_multipart_and_keeps_key_out_of_payload():
    provider = OCRSpaceProvider("https://api.ocr.space/parse/image", "test-secret")
    request = provider.build_request(b"synthetic-image")
    assert request.get_method() == "POST"
    assert request.headers["Apikey"] == "test-secret"
    assert request.headers["Content-type"].startswith("multipart/form-data; boundary=")
    for expected in (
        b'name="file"; filename="plate.jpg"',
        b'name="language"\r\n\r\neng',
        b'name="isOverlayRequired"\r\n\r\nfalse',
        b'name="scale"\r\n\r\ntrue',
        b'name="detectOrientation"\r\n\r\ntrue',
        b'name="OCREngine"\r\n\r\n2',
        b"synthetic-image",
    ):
        assert expected in request.data
    assert "test-secret" not in request.full_url
    assert b"test-secret" not in request.data


def test_ocr_space_normalizes_full_width_digits():
    backend = _ocr_space_backend(_ocr_space_response("１２３８"))
    assert backend.read(np.zeros((20, 60, 3), dtype=np.uint8)) == "1238"


def test_ocr_space_multiple_lines_prefer_plausible_lower_row():
    backend = _ocr_space_backend(_ocr_space_response("品川 500\n分類 12\n1238"))
    assert backend.read(np.zeros((20, 60, 3), dtype=np.uint8)) == "1238"
    assert backend.last_diagnostics.candidates == ("500", "1238")


def test_ocr_space_prefers_four_digits_across_variants():
    responses = iter((_ocr_space_response("238"), _ocr_space_response("1238")))

    def transport(_request, _timeout):
        return next(responses)

    backend = HTTPSOCRBackend(
        OCRSpaceProvider("https://api.ocr.space/parse/image", "test-secret"),
        transport=transport,
    )
    assert backend.read(np.zeros((20, 60, 3), dtype=np.uint8)) == "1238"
    assert backend.last_diagnostics.candidates == ("238", "1238")
    assert tuple(item.name for item in backend.last_diagnostics.variants) == (
        "full_plate",
        "lower_number_row",
    )


def test_ocr_space_joins_spaced_digits_without_inventing_digits():
    backend = _ocr_space_backend(_ocr_space_response("は 1 2 3 8"))
    assert backend.read(np.zeros((20, 60, 3), dtype=np.uint8)) == "1238"


def test_ocr_space_multiline_top_338_bottom_1238_selects_1238():
    backend = _ocr_space_backend(_ocr_space_response("品川 338\n分類\n1238"))
    assert backend.read(np.zeros((20, 60, 3), dtype=np.uint8)) == "1238"


def test_ocr_space_does_not_pad_three_digit_result():
    backend = _ocr_space_backend(_ocr_space_response("238"))
    assert backend.read(np.zeros((20, 60, 3), dtype=np.uint8)) == "238"


def test_ocr_space_preserves_leading_one():
    backend = _ocr_space_backend(_ocr_space_response("1238"))
    assert backend.read(np.zeros((20, 60, 3), dtype=np.uint8)) == "1238"


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError(),
        __import__("urllib.error").error.HTTPError(
            "https://redacted.invalid", 503, "unavailable", {}, None
        ),
    ],
    ids=["timeout", "provider-error"],
)
def test_ocr_space_network_failure_returns_empty_result(error):
    backend = _ocr_space_backend(error=error)
    assert backend.read(np.zeros((20, 60, 3), dtype=np.uint8)) == ""
    assert backend.last_diagnostics.raw_result == ""


def test_ocr_space_missing_key_fails_safely():
    backend = _ocr_space_backend(_ocr_space_response(), api_key="")
    assert backend.read(np.zeros((20, 60, 3), dtype=np.uint8)) == ""
    assert backend.last_diagnostics.attempt == "configuration-error"


def test_ocr_space_provider_error_fails_safely():
    backend = _ocr_space_backend(
        _ocr_space_response(
            "",
            ParsedResults=[],
            IsErroredOnProcessing=True,
            ErrorMessage=["Unable to recognize the file"],
        )
    )
    assert backend.read(np.zeros((20, 60, 3), dtype=np.uint8)) == ""
    assert backend.last_diagnostics.attempt == "provider-error"


@pytest.mark.parametrize("payload", [None, {}, {"ParsedResults": "bad"}])
def test_ocr_space_malformed_response_returns_empty_result(payload):
    backend = _ocr_space_backend(payload)
    assert backend.read(np.zeros((20, 60, 3), dtype=np.uint8)) == ""


def test_ocr_space_malformed_json_fails_safely():
    def transport(_request, _timeout):
        raise json.JSONDecodeError("bad JSON", "{", 0)

    backend = HTTPSOCRBackend(
        OCRSpaceProvider("https://api.ocr.space/parse/image", "test-secret"),
        transport=transport,
    )
    assert backend.read(np.zeros((20, 60, 3), dtype=np.uint8)) == ""
    assert backend.last_diagnostics.attempt == "malformed-response"


def test_ocr_space_success_without_digits_fails_safely():
    backend = _ocr_space_backend(_ocr_space_response("NO PLATE"))
    assert backend.read(np.zeros((20, 60, 3), dtype=np.uint8)) == ""
    assert backend.last_diagnostics.attempt == "no-digits"


def test_render_mode_is_lazy_and_does_not_load_or_run_local_ocr(
    monkeypatch, tmp_path
):
    sys.modules.pop("easyocr", None)
    sys.modules.pop("torch", None)
    process_calls = []
    monkeypatch.setattr(
        "anpr_web.ocr.subprocess.run",
        lambda *_args, **_kwargs: process_calls.append(True),
    )
    app = create_app(
        {
            "TESTING": True,
            "SQLITE_PATH": str(tmp_path / "cloud.sqlite3"),
            "ANPR_WEB_OCR_BACKEND": "ocr-space",
            "ANPR_OCR_API_URL": "https://api.ocr.space/parse/image",
            "ANPR_OCR_API_KEY": "never-render-this-key",
        }
    )
    client = app.test_client()
    for path in ("/health", "/", "/architecture"):
        response = client.get(path)
        assert response.status_code == 200
        assert b"never-render-this-key" not in response.data
    assert "easyocr" not in sys.modules
    assert "torch" not in sys.modules
    assert process_calls == []
    assert app.extensions["anpr_processor"]._backend is None


def test_tesseract_extracts_longest_mocked_digit_run(monkeypatch):
    completed = SimpleNamespace(returncode=0, stdout=b"12 3456\n")
    monkeypatch.setattr(
        "anpr_web.ocr.subprocess.run", lambda *args, **kwargs: completed
    )
    image = np.zeros((40, 120, 3), dtype=np.uint8)
    assert TesseractOCRBackend().read(image) == "3456"


def test_tesseract_never_exceeds_two_calls(monkeypatch):
    image = np.zeros((40, 120, 3), dtype=np.uint8)
    calls = []
    monkeypatch.setattr(
        "anpr_web.ocr._run_tesseract",
        lambda candidate, psm, timeout: calls.append((psm, timeout)) or "",
    )

    assert TesseractOCRBackend().read(image) == ""
    assert len(calls) == 2
    assert [psm for psm, _timeout in calls] == [7, 13]
    assert all(0 < timeout <= 3.0 for _psm, timeout in calls)


def test_tesseract_early_success_stops_later_calls(monkeypatch):
    calls = []

    def fake_run(_candidate, _psm, timeout):
        calls.append(timeout)
        return "1238"

    monkeypatch.setattr("anpr_web.ocr._run_tesseract", fake_run)
    image = np.zeros((40, 120, 3), dtype=np.uint8)
    assert TesseractOCRBackend().read(image) == "1238"
    assert len(calls) == 1


def test_tesseract_fallback_runs_only_after_empty_primary(monkeypatch):
    calls = []

    def fake_run(_candidate, psm, timeout):
        del timeout
        calls.append(psm)
        return "" if psm == 7 else "1238"

    monkeypatch.setattr("anpr_web.ocr._run_tesseract", fake_run)
    backend = TesseractOCRBackend()
    image = np.zeros((40, 120, 3), dtype=np.uint8)
    assert backend.read(image) == "1238"
    assert calls == [7, 13]
    assert backend.last_diagnostics.attempt == "fallback"
    assert backend.last_diagnostics.raw_result == "1238"


def test_total_ocr_deadline_stops_before_another_call(monkeypatch):
    now = [0.0]
    calls = []

    def fake_run(_candidate, _psm, timeout):
        calls.append(timeout)
        now[0] += 3.3
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


def test_exact_registered_1238_is_allowed(tmp_path):
    processor, image_path = _blocking_processor(
        tmp_path, lambda *_args, **_kwargs: "1238"
    )
    database = {
        "1238": {
            "name": "Synthetic Driver",
            "id": "DEMO-1",
            "type": "visitor",
        }
    }
    result = processor.process(image_path, database)
    assert result.status == "ALLOWED"
    assert result.match["matched_plate"] == "1238"
    assert result.match["distance"] == 0


def test_close_up_fallback_ocr_uses_complete_upload(tmp_path):
    image_path = tmp_path / "close-up.png"
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    frame[:, :10] = 255
    cv2.imwrite(str(image_path), frame)
    observed = []
    runtime = SimpleNamespace(
        ocr_confidence_threshold=0.5,
        min_ocr_length=3,
        matching_policy="exact",
        match_tolerance=0,
    )

    def detector(upload):
        return upload[:, 10:70], (10, 0, 60, 40), "Close-up fallback"

    def ocr(source, *_args, **_kwargs):
        observed.append(source.copy())
        return "238"

    WebProcessor(runtime, detector=detector, ocr=ocr).process(image_path, {})
    assert observed[0].shape == frame.shape
    assert np.all(observed[0][:, :10] == 255)


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
    unknown = WebProcessor(runtime, detector=detector, ocr=lambda *_args, **_kw: "1236")
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
