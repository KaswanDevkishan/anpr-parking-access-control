"""Lazy, configurable OCR adapters used only by the Flask application."""

import importlib
import re
import subprocess
import unicodedata
from collections import Counter
from dataclasses import dataclass

import cv2

DIGIT_PSMS = (6, 7, 8, 10, 13)
UPSCALE_FACTORS = (2, 3, 4)


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
    """Vote across lightweight, independent Tesseract digit observations."""

    # Retained as a public description of the sequence modes for diagnostics/tests.
    PASSES = tuple(
        (preprocessing, psm)
        for preprocessing in (
            "grayscale",
            "otsu",
            "inverted_otsu",
            "adaptive",
            "morphology",
        )
        for psm in DIGIT_PSMS
    )

    def __init__(self):
        self.last_diagnostics = None

    def read(self, plate_image, confidence_threshold=0.5):
        del confidence_threshold  # Tesseract CLI has no comparable confidence input.
        observations = []
        for crop_name, crop in _plate_crops(plate_image):
            for variant_name, variant in _preprocessing_variants(crop):
                for psm in DIGIT_PSMS:
                    for sequence in _plausible_sequences(_run_tesseract(variant, psm)):
                        observations.append(
                            OCRCandidate(
                                digits=sequence,
                                crop=crop_name,
                                preprocessing=variant_name,
                                psm=psm,
                            )
                        )

        sequence_candidates = {item.digits for item in observations}
        if len(sequence_candidates) > 1:
            segmented = _segmented_digit_candidate(plate_image)
            if segmented and len(segmented) == 4:
                observations.append(
                    OCRCandidate(
                        digits=segmented,
                        crop="segmented_bottom_row",
                        preprocessing="individual_characters",
                        psm=10,
                    )
                )

        selected, uncertain, votes = _select_candidate(observations)
        self.last_diagnostics = OCRDiagnostics(
            backend="tesseract",
            selected=selected,
            candidates=tuple(observations),
            vote_counts=tuple(
                sorted(votes.items(), key=lambda item: (-item[1], item[0]))
            ),
            uncertain=uncertain,
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
    vote_counts: tuple[tuple[str, int], ...] = ()
    uncertain: bool = False

    @property
    def alternatives(self):
        selected_digits = self.selected.digits if self.selected else None
        return tuple(
            (digits, count)
            for digits, count in self.vote_counts
            if digits != selected_digits
        )


def _plate_crops(plate_image):
    """Yield several tight variants around the plate's large bottom row."""
    height, width = plate_image.shape[:2]
    if height == 0 or width == 0:
        return
    # The vertical offsets deliberately move slightly above/below the expected
    # baseline while side insets suppress the border and mounting bolts.
    crop_specs = (
        ("lower_balanced", 0.42, 0.94, 0.07, 0.93),
        ("lower_high", 0.36, 0.90, 0.08, 0.92),
        ("lower_low", 0.48, 0.98, 0.06, 0.94),
    )
    for name, top, bottom, left, right in crop_specs:
        crop = plate_image[
            int(height * top) : max(int(height * top) + 1, int(height * bottom)),
            int(width * left) : max(int(width * left) + 1, int(width * right)),
        ]
        if crop.size:
            yield name, crop


def _preprocessing_variants(plate_image):
    """Yield scaled grayscale and threshold variants without retaining a batch."""
    gray = (
        plate_image
        if plate_image.ndim == 2
        else cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    )
    for factor in UPSCALE_FACTORS:
        enlarged = cv2.resize(
            gray,
            None,
            fx=factor,
            fy=factor,
            interpolation=cv2.INTER_CUBIC,
        )
        otsu = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        inverted = cv2.bitwise_not(otsu)
        block_size = min(31, min(enlarged.shape[:2]) // 2 * 2 - 1)
        yield f"grayscale_{factor}x", enlarged
        yield f"otsu_{factor}x", otsu
        yield f"inverted_otsu_{factor}x", inverted
        if block_size >= 3:
            adaptive = cv2.adaptiveThreshold(
                enlarged,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                7,
            )
            yield f"adaptive_{factor}x", adaptive
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morphology = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        yield f"morphology_{factor}x", morphology


def _segmented_digit_candidate(plate_image):
    """Locate four bottom-row glyphs and OCR them independently as one vote."""
    best_regions = ()
    best_crop = None
    for _crop_name, crop in _plate_crops(plate_image):
        gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[
            1
        ]
        contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
            0
        ]
        height, width = binary.shape[:2]
        regions = []
        for contour in contours:
            x, y, region_width, region_height = cv2.boundingRect(contour)
            if (
                region_height >= height * 0.38
                and region_width >= width * 0.025
                and region_width <= width * 0.28
                and region_height > region_width
            ):
                regions.append((x, y, region_width, region_height))
        regions.sort()
        if len(regions) == 4:
            best_regions, best_crop = regions, gray
            break
    if best_crop is None:
        return ""

    digits = []
    for x, y, width, height in best_regions:
        # Coordinates came from the 3x image.
        x0, y0 = max(0, x // 3 - 2), max(0, y // 3 - 2)
        x1 = min(best_crop.shape[1], (x + width) // 3 + 2)
        y1 = min(best_crop.shape[0], (y + height) // 3 + 2)
        glyph = best_crop[y0:y1, x0:x1]
        glyph = cv2.copyMakeBorder(glyph, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
        glyph = cv2.resize(glyph, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        results = _digit_sequences(_run_tesseract(glyph, 10))
        single_digits = [result for result in results if len(result) == 1]
        if not single_digits:
            return ""
        digits.append(single_digits[0])
    return "".join(digits)


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


def _plausible_sequences(text):
    """Retain only complete, plausible large-row identifiers."""
    return tuple(
        sequence for sequence in _digit_sequences(text) if len(sequence) in (3, 4)
    )


def _select_candidate(candidates):
    """Aggregate independent observations and reject unresolved leading ties."""
    # A particular crop/preprocessing/PSM result counts at most once per number.
    evidence = {
        (item.digits, item.crop, item.preprocessing, item.psm) for item in candidates
    }
    votes = Counter(digits for digits, *_source in evidence)
    if not votes:
        return None, False, votes

    ranked = sorted(votes, key=lambda digits: (-votes[digits], -len(digits), digits))
    top_support = votes[ranked[0]]
    comparable = [digits for digits in ranked if top_support - votes[digits] <= 1]
    four_digit = [digits for digits in comparable if len(digits) == 4]
    preferred = four_digit or comparable
    best_support = max(votes[digits] for digits in preferred)
    leaders = [digits for digits in preferred if votes[digits] == best_support]
    selected_digits = leaders[0]
    selected = next(item for item in candidates if item.digits == selected_digits)
    # Different same-length readings at equal leading support are genuinely
    # ambiguous. A four-digit reading wins over a comparable three-digit one.
    uncertain = len(leaders) > 1
    return selected, uncertain, votes
