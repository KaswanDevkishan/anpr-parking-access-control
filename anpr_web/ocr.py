"""Lazy, configurable OCR adapters used only by the Flask application."""

import importlib
import re
import subprocess

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

    def read(self, plate_image, confidence_threshold=0.5):
        del confidence_threshold  # Tesseract's CLI does not expose segment confidence.
        processed = _preprocess_plate(plate_image)
        encoded, payload = cv2.imencode(".png", processed)
        if not encoded:
            return ""
        completed = subprocess.run(
            [
                "tesseract",
                "stdin",
                "stdout",
                "--psm",
                "7",
                "-l",
                "eng",
                "-c",
                "tessedit_char_whitelist=0123456789",
            ],
            input=payload.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=20,
        )
        if completed.returncode != 0:
            return ""
        text = completed.stdout.decode("utf-8", errors="ignore")
        numbers = re.findall(r"\d+", text)
        return max(numbers, key=len) if numbers else ""


def _preprocess_plate(plate_image):
    """Reuse the local OCR workflow's resize, denoise, and threshold steps."""
    height, width = plate_image.shape[:2]
    scale = 100.0 / height
    resized = cv2.resize(plate_image, (max(1, int(width * scale)), 100))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    return cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
