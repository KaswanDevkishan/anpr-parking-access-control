"""Lazy, configurable OCR adapters used only by the Flask application."""

import importlib
import re
import subprocess
import unicodedata
from dataclasses import dataclass

import cv2


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
    """Small subprocess adapter for Tesseract's digit-only recognition."""

    PASSES = (
        ("grayscale", 6),
        ("grayscale", 11),
        ("enlarged", 7),
        ("otsu", 7),
        ("otsu", 8),
        ("otsu", 13),
        ("inverted_otsu", 7),
        ("inverted_otsu", 8),
        ("inverted_otsu", 13),
        ("adaptive", 6),
        ("adaptive", 11),
    )

    def __init__(self):
        self.last_diagnostics = None

    def read(self, plate_image, confidence_threshold=0.5):
        del confidence_threshold  # Tesseract's CLI does not expose segment confidence.
        candidates = []
        for crop_name, crop in _plate_crops(plate_image):
            variants = dict(_preprocessing_variants(crop))
            for variant_name, psm in self.PASSES:
                if variant_name not in variants:
                    continue
                raw_text = _run_tesseract(variants[variant_name], psm)
                for sequence in _digit_sequences(raw_text):
                    candidates.append(
                        OCRCandidate(
                            digits=sequence,
                            crop=crop_name,
                            preprocessing=variant_name,
                            psm=psm,
                        )
                    )
                # A four-digit lower-row result is already the strongest class.
                if crop_name != "full" and any(
                    item.crop == crop_name and len(item.digits) == 4
                    for item in candidates
                ):
                    break
            if crop_name == "lower_55_percent" and any(
                item.crop == crop_name and len(item.digits) in (3, 4)
                for item in candidates
            ):
                break

        selected = max(candidates, key=_candidate_score, default=None)
        self.last_diagnostics = OCRDiagnostics(
            backend="tesseract",
            selected=selected,
            candidates=tuple(candidates),
        )
        return selected.digits if selected is not None else ""


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


def _plate_crops(plate_image):
    """Yield the whole plate and regions biased toward its large bottom row."""
    height, width = plate_image.shape[:2]
    if height == 0 or width == 0:
        return
    yield "full", plate_image
    yield "lower_55_percent", plate_image[int(height * 0.45) :, :]
    yield (
        "centered_lower",
        plate_image[
            int(height * 0.40) : int(height * 0.94),
            int(width * 0.08) : int(width * 0.92),
        ],
    )


def _preprocessing_variants(plate_image):
    """Create conservative variants without eroding the plate's thick digits."""
    gray = (
        plate_image
        if plate_image.ndim == 2
        else cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    )
    height, width = gray.shape[:2]
    scale = max(1.0, 120.0 / height)
    enlarged = cv2.resize(
        gray,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    otsu = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    inverted_otsu = cv2.bitwise_not(otsu)
    block_size = min(31, min(enlarged.shape[:2]) // 2 * 2 - 1)
    variants = [
        ("grayscale", gray),
        ("enlarged", enlarged),
        ("otsu", otsu),
        ("inverted_otsu", inverted_otsu),
    ]
    if block_size >= 3:
        variants.append(
            (
                "adaptive",
                cv2.adaptiveThreshold(
                    enlarged,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    block_size,
                    7,
                ),
            )
        )
    return variants


def _run_tesseract(image, psm):
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
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired):
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


def _candidate_score(candidate):
    """Prefer a plausible large bottom-row identifier over classification text."""
    length = len(candidate.digits)
    plausible = length in (3, 4)
    return (
        plausible,
        length == 4,
        candidate.crop != "full",
        candidate.crop == "centered_lower",
        min(length, 4),
    )
