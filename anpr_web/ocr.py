"""Lazy, configurable OCR adapters used only by the Flask application."""

import importlib
import json
import re
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import cv2

MAX_TESSERACT_CALLS = 2
MAX_OCR_SPACE_CALLS = 2
OCR_BUDGET_SECONDS = 6.5
TESSERACT_TIMEOUT_SECONDS = 3.0


class OCRTimeout(RuntimeError):
    """Raised when a bounded OCR scan runs out of time."""


def create_web_ocr_backend(
    name,
    *,
    api_url="",
    api_key="",
    timeout_seconds=5.0,
    transport=None,
):
    """Return an OCR adapter without importing either OCR implementation."""
    normalized = (name or "").strip().lower()
    if normalized == "ocr-space":
        return HTTPSOCRBackend(
            OCRSpaceProvider(api_url, api_key),
            timeout_seconds=timeout_seconds,
            transport=transport,
        )
    if normalized == "tesseract":
        return TesseractOCRBackend()
    if normalized == "easyocr":
        return EasyOCRBackend()
    raise ValueError(
        "ANPR_WEB_OCR_BACKEND must be 'ocr-space', 'easyocr', or 'tesseract'"
    )


class HTTPSOCRBackend:
    """Provider-neutral HTTPS OCR client with a bounded, fail-closed request."""

    def __init__(
        self,
        provider,
        *,
        timeout_seconds=5.0,
        transport=None,
        clock=time.monotonic,
    ):
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self._transport = transport or _send_json
        self._clock = clock
        self.last_diagnostics = None

    def read(self, plate_image, confidence_threshold=0.5):
        del confidence_threshold
        started = self._clock()
        candidates = []
        variants = []
        selected = ""
        status = "request-failed"
        try:
            deadline = started + self.timeout_seconds
            for variant_name, variant_image in _ocr_space_variants(plate_image)[
                :MAX_OCR_SPACE_CALLS
            ]:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    status = "timeout"
                    break
                encoded, payload = cv2.imencode(".jpg", variant_image)
                if not encoded:
                    variants.append(
                        OCRVariantDiagnostics(
                            variant_name, "", (), "image-encoding-failed"
                        )
                    )
                    continue
                try:
                    request = self.provider.build_request(payload.tobytes())
                    response = self._transport(request, remaining)
                    response_candidates = self.provider.parse_response(response)
                    raw_text = self.provider.sanitized_text(response)
                    candidates.extend(response_candidates)
                    variants.append(
                        OCRVariantDiagnostics(
                            variant_name,
                            raw_text,
                            tuple(item.text for item in response_candidates),
                            "success",
                        )
                    )
                except (TimeoutError, OCRTimeout):
                    variants.append(
                        OCRVariantDiagnostics(variant_name, "", (), "timeout")
                    )
                    status = "timeout"
                    break
                except HTTPError:
                    variants.append(
                        OCRVariantDiagnostics(variant_name, "", (), "http-error")
                    )
                except OCRProviderError:
                    variants.append(
                        OCRVariantDiagnostics(variant_name, "", (), "provider-error")
                    )
                except (
                    json.JSONDecodeError,
                    TypeError,
                    AttributeError,
                    KeyError,
                    IndexError,
                ):
                    variants.append(
                        OCRVariantDiagnostics(
                            variant_name, "", (), "malformed-response"
                        )
                    )
                except (URLError, OSError):
                    variants.append(
                        OCRVariantDiagnostics(variant_name, "", (), "network-error")
                    )
            selected = _select_plate_candidate(candidates)
            if selected:
                status = "success"
            elif status != "timeout":
                status = (
                    variants[-1].status
                    if variants and all(item.status != "success" for item in variants)
                    else "no-digits"
                )
            return selected
        except ValueError:
            status = "configuration-error"
            return ""
        finally:
            self.last_diagnostics = OCRDiagnostics(
                backend=self.provider.name,
                attempt=status,
                elapsed_seconds=max(0.0, self._clock() - started),
                raw_result=selected,
                candidates=tuple(
                    dict.fromkeys(
                        item.text for item in candidates if len(item.text) <= 16
                    )
                )[:10],
                variants=tuple(variants),
            )


@dataclass(frozen=True)
class OCRCandidate:
    """Provider-neutral recognized text with optional image geometry."""

    text: str
    top: float | None = None
    bottom: float | None = None
    area: float = 0.0


class OCRProviderError(ValueError):
    """Raised for a valid provider response that reports processing failure."""


class OCRSpaceProvider:
    """Translate between the neutral OCR client and OCR.Space's multipart API."""

    name = "OCR.Space"

    def __init__(self, api_url, api_key):
        self.api_url = (api_url or "").strip()
        self.api_key = api_key or ""

    def build_request(self, image_bytes):
        if not self.api_url.startswith("https://") or not self.api_key:
            raise ValueError("Cloud OCR HTTPS configuration is incomplete.")
        boundary = f"anpr-{uuid4().hex}"
        body = _multipart_body(
            boundary,
            (
                ("language", "eng"),
                ("isOverlayRequired", "false"),
                ("scale", "true"),
                ("detectOrientation", "true"),
                ("OCREngine", "2"),
            ),
            "file",
            "plate.jpg",
            "image/jpeg",
            image_bytes,
        )
        return Request(
            self.api_url,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "apikey": self.api_key,
            },
            method="POST",
        )

    def parse_response(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Malformed OCR response.")
        if payload.get("IsErroredOnProcessing") is True or payload.get("ErrorMessage"):
            raise OCRProviderError("OCR provider could not process the image.")
        parsed_results = payload.get("ParsedResults")
        if not isinstance(parsed_results, list) or not parsed_results:
            raise OCRProviderError("OCR provider returned no parsed results.")
        candidates = []
        line_number = 0
        for result in parsed_results:
            if not isinstance(result, dict) or not isinstance(
                result.get("ParsedText"), str
            ):
                raise ValueError("Malformed OCR response.")
            for line in result["ParsedText"].splitlines() or [result["ParsedText"]]:
                for sequence in _digit_sequences(line):
                    candidates.append(
                        OCRCandidate(sequence, line_number, line_number, len(sequence))
                    )
                line_number += 1
        return tuple(candidates)

    def sanitized_text(self, payload):
        """Return bounded OCR text suitable for authenticated diagnostics."""
        parsed_results = payload.get("ParsedResults", [])
        text = "\n".join(
            result.get("ParsedText", "")
            for result in parsed_results
            if isinstance(result, dict)
        )
        text = "".join(
            character if character.isprintable() or character == "\n" else " "
            for character in text
        )
        return text.strip()[:500]


def _multipart_body(boundary, fields, file_field, filename, content_type, content):
    """Build one multipart request body without retaining or logging the image."""
    marker = boundary.encode("ascii")
    body = bytearray()
    for name, value in fields:
        body.extend(b"--" + marker + b"\r\n")
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode(
                "ascii"
            )
        )
    body.extend(b"--" + marker + b"\r\n")
    body.extend(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'
        ).encode("ascii")
    )
    body.extend(content)
    body.extend(b"\r\n--" + marker + b"--\r\n")
    return bytes(body)


def _send_json(request, timeout_seconds):
    """Send JSON without logging the URL, credentials, image, or response body."""
    with urlopen(request, timeout=timeout_seconds) as response:
        if not 200 <= response.status < 300:
            raise HTTPError(request.full_url, response.status, "OCR error", {}, None)
        return json.loads(response.read())


def _select_plate_candidate(candidates):
    plausible = [candidate for candidate in candidates if len(candidate.text) in (3, 4)]
    if not plausible:
        return ""
    # Japanese lower rows commonly contain four digits. Prefer an observed
    # four-digit sequence, without synthesizing or padding a shorter reading.
    return max(
        plausible,
        key=lambda candidate: (
            len(candidate.text) == 4,
            candidate.bottom is not None,
            candidate.bottom if candidate.bottom is not None else -1,
            candidate.area,
        ),
    ).text


class EasyOCRBackend:
    """Lazy bridge to the unchanged Raspberry Pi EasyOCR module."""

    def __init__(self):
        self._engine = None
        self._reader = None

    def read(self, plate_image, confidence_threshold=0.5):
        if self._engine is None:
            self._engine = importlib.import_module("ocr_engine")
        if self._reader is None:
            self._reader = self._engine.load_ocr()
        return self._engine.read_plate(
            plate_image,
            self._reader,
            confidence_threshold=confidence_threshold,
        )


class TesseractOCRBackend:
    """Run one primary pass and, only when empty, one bounded fallback pass."""

    PASSES = (
        ("primary", "otsu_3x", 7),
        ("fallback", "inverted_otsu_3x", 13),
    )

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self.last_diagnostics = None

    def read(self, plate_image, confidence_threshold=0.5):
        del confidence_threshold  # Tesseract CLI has no comparable confidence input.
        started = self._clock()
        deadline = started + OCR_BUDGET_SECONDS

        for crop_name, preprocessing, psm in self.PASSES[:MAX_TESSERACT_CALLS]:
            remaining = deadline - self._clock()
            if remaining <= 0:
                self._record_diagnostics(started, None, "")
                raise OCRTimeout("The total OCR deadline was exhausted.")

            candidate_image = _prepare_candidate(plate_image, crop_name, preprocessing)
            try:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    self._record_diagnostics(started, None, "")
                    raise OCRTimeout("The total OCR deadline was exhausted.")
                try:
                    text = _run_tesseract(
                        candidate_image,
                        psm,
                        timeout=min(TESSERACT_TIMEOUT_SECONDS, remaining),
                    )
                except OCRTimeout:
                    self._record_diagnostics(started, crop_name, "")
                    raise
            finally:
                # Do not retain transformed image arrays between passes.
                del candidate_image

            if self._clock() > deadline:
                self._record_diagnostics(started, crop_name, "")
                raise OCRTimeout("The total OCR deadline was exhausted.")

            sequences = _plausible_sequences(text)
            if sequences:
                selected_digits = max(sequences, key=len)
                self._record_diagnostics(started, crop_name, selected_digits)
                return selected_digits

        self._record_diagnostics(started, "fallback", "")
        return ""

    def _record_diagnostics(self, started, attempt, raw_result):
        self.last_diagnostics = OCRDiagnostics(
            backend="tesseract",
            attempt=attempt,
            elapsed_seconds=max(0.0, self._clock() - started),
            raw_result=raw_result,
        )


@dataclass(frozen=True)
class OCRDiagnostics:
    """Structured details kept on the backend and not rendered publicly."""

    backend: str
    attempt: str | None
    elapsed_seconds: float
    raw_result: str
    uncertain: bool = False
    candidates: tuple[str, ...] = ()
    variants: tuple["OCRVariantDiagnostics", ...] = ()


@dataclass(frozen=True)
class OCRVariantDiagnostics:
    """Sanitized details for one bounded OCR.Space request."""

    name: str
    raw_text: str
    candidates: tuple[str, ...]
    status: str


def _ocr_space_variants(plate_image):
    """Build the two lightweight OCR.Space inputs without retaining either."""
    height, width = plate_image.shape[:2]
    full = cv2.resize(
        plate_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
    )
    lower_top = int(height * 0.40)
    lower = plate_image[lower_top:height, 0:width]
    gray = lower if lower.ndim == 2 else cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    enlarged = cv2.resize(
        enhanced, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
    )
    return (("full_2x", full), ("lower_60pct_contrast_3x", enlarged))


def _prepare_candidate(plate_image, crop_name, preprocessing):
    """Create exactly one stage image, avoiding a transformed-image batch."""
    height, width = plate_image.shape[:2]
    if not height or not width:
        return plate_image

    if crop_name == "fallback":
        top, bottom, left, right = 0.42, 0.98, 0.15, 0.97
    else:
        top, bottom, left, right = 0.50, 0.96, 0.24, 0.95
    crop = plate_image[
        int(height * top) : max(int(height * top) + 1, int(height * bottom)),
        int(width * left) : max(int(width * left) + 1, int(width * right)),
    ]
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    if preprocessing == "grayscale_3x":
        return enlarged
    threshold_type = cv2.THRESH_BINARY
    if preprocessing == "inverted_otsu_3x":
        threshold_type = cv2.THRESH_BINARY_INV
    return cv2.threshold(enlarged, 0, 255, threshold_type + cv2.THRESH_OTSU)[1]


def _run_tesseract(image, psm, timeout):
    """Run one in-memory Tesseract process with a caller-supplied hard timeout."""
    encoded, payload = cv2.imencode(".png", image)
    if not encoded:
        return ""
    try:
        completed = subprocess.run(
            [
                "tesseract",
                "stdin",
                "stdout",
                "--psm",
                str(psm),
                "-l",
                "eng",
                "-c",
                "tessedit_char_whitelist=0123456789",
            ],
            input=payload.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        # subprocess.run kills and waits for the child before raising.
        raise OCRTimeout("A Tesseract subprocess timed out.") from exc
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.decode("utf-8", errors="ignore")


def _digit_sequences(text):
    normalized = "".join(
        str(unicodedata.digit(character))
        if character.isdigit() and not character.isascii()
        else character
        for character in text
    )
    return re.findall(r"\d+", normalized)


def _plausible_sequences(text):
    """Retain only complete, plausible large-row identifiers."""
    return tuple(
        sequence for sequence in _digit_sequences(text) if len(sequence) in (3, 4)
    )
