import json
import sys
from email.message import EmailMessage
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from anpr_web import create_app
from anpr_web.mail import HTTPAPIBackend, SMTPBackend
from anpr_web.ocr import (
    EasyOCRBackend,
    OCRCandidate,
    TesseractOCRBackend,
    _select_candidate,
)
from anpr_web.processing import WebProcessor


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


def _mock_tesseract_pipeline(monkeypatch, outputs):
    monkeypatch.setattr(
        "anpr_web.ocr._plate_crops",
        lambda image: (("full", image), ("lower_55_percent", image)),
    )
    monkeypatch.setattr(
        "anpr_web.ocr._preprocessing_variants",
        lambda image: (
            (name, np.full((2, 2), index, dtype=np.uint8))
            for index, name in enumerate(
                ("grayscale", "enlarged", "otsu", "inverted_otsu", "adaptive")
            )
        ),
    )
    crop_calls = {"count": 0}

    def fake_run(image, psm):
        pass_index = crop_calls["count"]
        crop_calls["count"] += 1
        crop = "full" if pass_index < len(TesseractOCRBackend.PASSES) else "lower"
        variant = (
            "grayscale",
            "enlarged",
            "otsu",
            "inverted_otsu",
            "adaptive",
        )[int(image[0, 0])]
        return outputs.get((crop, variant, psm), "")

    monkeypatch.setattr("anpr_web.ocr._run_tesseract", fake_run)


def test_tesseract_uses_lower_crop_when_full_image_is_empty(monkeypatch):
    _mock_tesseract_pipeline(monkeypatch, {("lower", "grayscale", 6): "1238"})
    backend = TesseractOCRBackend()
    assert backend.read(np.zeros((40, 120, 3), dtype=np.uint8)) == "1238"
    assert backend.last_diagnostics.selected.crop == "lower_55_percent"


def test_tesseract_prefers_four_digit_lower_row_over_top_classification(monkeypatch):
    _mock_tesseract_pipeline(
        monkeypatch,
        {
            ("full", "grayscale", 6): "338",
            ("lower", "grayscale", 6): "1238",
        },
    )
    assert TesseractOCRBackend().read(np.zeros((40, 120, 3), dtype=np.uint8)) == "1238"


def test_tesseract_inverted_threshold_can_succeed(monkeypatch):
    _mock_tesseract_pipeline(monkeypatch, {("lower", "inverted_otsu", 8): "1238"})
    backend = TesseractOCRBackend()
    assert backend.read(np.zeros((40, 120, 3), dtype=np.uint8)) == "1238"
    assert backend.last_diagnostics.selected.preprocessing == "inverted_otsu"


def test_tesseract_tries_multiple_page_segmentation_modes(monkeypatch):
    _mock_tesseract_pipeline(monkeypatch, {("lower", "otsu", 13): "1238"})
    backend = TesseractOCRBackend()
    assert backend.read(np.zeros((40, 120, 3), dtype=np.uint8)) == "1238"
    assert backend.last_diagnostics.selected.psm == 13


def test_tesseract_no_usable_digits_stays_empty(monkeypatch):
    _mock_tesseract_pipeline(monkeypatch, {})
    assert TesseractOCRBackend().read(np.zeros((40, 120, 3), dtype=np.uint8)) == ""


def candidate(digits, variant, crop="lower_balanced", psm=7):
    return OCRCandidate(digits, crop, variant, psm)


def test_candidate_voting_selects_1238_from_several_variants():
    selected, uncertain, votes = _select_candidate(
        [
            candidate("1238", "grayscale_2x"),
            candidate("1238", "otsu_3x"),
            candidate("1238", "adaptive_4x", psm=13),
            candidate("1236", "inverted_otsu_3x"),
        ]
    )
    assert selected.digits == "1238"
    assert votes == {"1238": 3, "1236": 1}
    assert not uncertain


def test_candidate_voting_returns_uncertain_for_equal_support():
    selected, uncertain, votes = _select_candidate(
        [
            candidate("1236", "grayscale_2x"),
            candidate("1238", "otsu_3x"),
        ]
    )
    assert selected.digits in {"1236", "1238"}
    assert votes == {"1236": 1, "1238": 1}
    assert uncertain


def test_segmented_digit_fallback_is_counted_as_an_independent_vote(monkeypatch):
    image = np.zeros((40, 120, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "anpr_web.ocr._plate_crops", lambda _image: (("lower_balanced", image),)
    )
    variants = (
        ("grayscale_2x", np.zeros((2, 2), dtype=np.uint8)),
        ("otsu_3x", np.ones((2, 2), dtype=np.uint8)),
    )
    monkeypatch.setattr("anpr_web.ocr._preprocessing_variants", lambda _image: variants)
    monkeypatch.setattr(
        "anpr_web.ocr._run_tesseract",
        lambda variant, psm: (
            "1238"
            if (variant[0, 0] == 0 and psm == 6)
            else "1236"
            if (variant[0, 0] == 1 and psm == 7)
            else ""
        ),
    )
    monkeypatch.setattr(
        "anpr_web.ocr._segmented_digit_candidate", lambda _image: "1238"
    )

    backend = TesseractOCRBackend()
    assert backend.read(image) == "1238"
    assert dict(backend.last_diagnostics.vote_counts) == {"1238": 2, "1236": 1}
    assert any(
        item.crop == "segmented_bottom_row"
        for item in backend.last_diagnostics.candidates
    )


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
