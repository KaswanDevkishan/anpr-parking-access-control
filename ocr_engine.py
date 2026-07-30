"""
ocr_engine.py
─────────────────────────────────────────────────────────────
Reads text from a cropped licence plate image using EasyOCR.
"""

import re

import cv2
import easyocr

# ── Module-level singleton: created once, reused every call ──────────────────
_reader = None


def load_ocr():
    """
    Initialise the EasyOCR reader (Japanese + English).
    """
    global _reader
    if _reader is None:
        print("   Initialising EasyOCR …")
        print("   (First run downloads language models — ~200 MB, takes a minute)")
        _reader = easyocr.Reader(["ja", "en"], gpu=False)
        print("   EasyOCR ready ✅")
    return _reader


def _preprocess_plate(plate_img):
    """
    Enhance the cropped plate image so OCR works better.
    """
    h, w = plate_img.shape[:2]

    # Scale to 100 px tall, keep aspect ratio
    scale = 100.0 / h
    resized = cv2.resize(plate_img, (int(w * scale), 100))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Otsu: automatically picks the best threshold value
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def read_plate(plate_img, reader=None, confidence_threshold=0.5):
    """
    Run OCR on a cropped plate image and return the plate number string.
    """
    if reader is None:
        reader = load_ocr()

    processed = _preprocess_plate(plate_img)
    results = reader.readtext(processed)

    # Concatenate all recognised text segments (filter low-confidence junk)
    all_text = "".join(item[1] for item in results if item[2] > confidence_threshold)

    # Extract digit runs — the 4-digit number is the primary identifier
    numbers = re.findall(r"\d+", all_text)
    if numbers:
        return max(numbers, key=len)

    # Fallback: return everything
    return all_text.strip()
