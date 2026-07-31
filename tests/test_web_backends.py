import json
import sys
from email.message import EmailMessage
from types import SimpleNamespace

import numpy as np

from anpr_web import create_app
from anpr_web.mail import HTTPAPIBackend, SMTPBackend
from anpr_web.ocr import EasyOCRBackend, TesseractOCRBackend


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
