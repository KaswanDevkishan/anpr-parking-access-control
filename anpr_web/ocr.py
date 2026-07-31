"""Lazy, configurable OCR adapters used only by the Flask application."""

import importlib
import re
import subprocess
import time
import unicodedata
from dataclasses import dataclass

import cv2

MAX_TESSERACT_CALLS = 3
OCR_BUDGET_SECONDS = 4.0
TESSERACT_TIMEOUT_SECONDS = 1.2


class OCRTimeout(RuntimeError):
    """Raised when a bounded OCR scan runs out of time."""


def create_web_ocr_backend(name):
    """Return an OCR adapter without importing either OCR implementation."""
    normalized = (name or "").strip().lower()
    if normalized == "tesseract":
        return TesseractOCRBackend()
    if normalized == "easyocr":
        return EasyOCRBackend()
    raise ValueError("ANPR_WEB_OCR_BACKEND must be 'easyocr' or 'tesseract'")


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
    """Run at most three sequential Tesseract passes within a hard deadline."""

    PASSES = (
        ("lower_balanced", "otsu_3x", 7),
        ("lower_alternate", "inverted_otsu_3x", 7),
        ("lower_balanced", "grayscale_3x", 13),
    )

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self.last_diagnostics = None

    def read(self, plate_image, confidence_threshold=0.5):
        del confidence_threshold  # Tesseract CLI has no comparable confidence input.
        deadline = self._clock() + OCR_BUDGET_SECONDS
        observations = []

        for crop_name, preprocessing, psm in self.PASSES[:MAX_TESSERACT_CALLS]:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise OCRTimeout("The total OCR deadline was exhausted.")

            candidate_image = _prepare_candidate(plate_image, crop_name, preprocessing)
            try:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise OCRTimeout("The total OCR deadline was exhausted.")
                text = _run_tesseract(
                    candidate_image,
                    psm,
                    timeout=min(TESSERACT_TIMEOUT_SECONDS, remaining),
                )
            finally:
                # Do not retain transformed image arrays between passes.
                del candidate_image

            sequences = _plausible_sequences(text)
            if sequences:
                selected_digits = max(sequences, key=len)
                selected = OCRCandidate(
                    digits=selected_digits,
                    crop=crop_name,
                    preprocessing=preprocessing,
                    psm=psm,
                )
                observations.append(selected)
                self.last_diagnostics = OCRDiagnostics(
                    backend="tesseract",
                    selected=selected,
                    candidates=tuple(observations),
                )
                return selected_digits

        self.last_diagnostics = OCRDiagnostics(
            backend="tesseract",
            selected=None,
            candidates=tuple(observations),
        )
        return ""


@dataclass(frozen=True)
class OCRCandidate:
    """One normalized result retained for private/admin diagnostics."""

    digits: str
    crop: str
    preprocessing: str
    psm: int


@dataclass(frozen=True)
class OCRDiagnostics:
    """Structured details kept on the backend and not rendered publicly."""

    backend: str
    selected: OCRCandidate | None
    candidates: tuple[OCRCandidate, ...]
    vote_counts: tuple[tuple[str, int], ...] = ()
    uncertain: bool = False

    @property
    def alternatives(self):
        return ()


def _prepare_candidate(plate_image, crop_name, preprocessing):
    """Create exactly one stage image, avoiding a transformed-image batch."""
    height, width = plate_image.shape[:2]
    if not height or not width:
        return plate_image

    if crop_name == "lower_alternate":
        top, bottom, left, right = 0.48, 0.98, 0.06, 0.94
    else:
        top, bottom, left, right = 0.42, 0.94, 0.07, 0.93
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
