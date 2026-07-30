"""Headless uploaded-image processing built from the existing ANPR modules."""

import base64
import os
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from matcher import find_match
from ocr_engine import load_ocr, read_plate
from plate_detector import detect_plate_for_upload

ALLOWED_FORMATS = {"JPEG": ".jpg", "PNG": ".png"}


class InvalidImage(ValueError):
    """Raised when an upload is not a supported, decodable image."""


@dataclass(frozen=True)
class ProcessingResult:
    """Presentation-safe result from one uploaded image."""

    status: str
    reason: str
    ocr_text: str
    match: dict | None
    annotated_image: str
    detection_method: str | None = None


@dataclass(frozen=True)
class ImageReview:
    """Temporary presentation data for image-assisted registration."""

    original_image: str
    cropped_image: str | None
    ocr_text: str
    detection_method: str | None
    error: str | None


class WebProcessor:
    """Run the existing detector, OCR, and matcher in a headless pipeline."""

    def __init__(
        self,
        runtime_config,
        detector=detect_plate_for_upload,
        ocr=read_plate,
        ocr_loader=load_ocr,
    ):
        self.config = runtime_config
        self.detector = detector
        self.ocr = ocr
        self.ocr_loader = ocr_loader
        self._reader = None

    def process(self, image_path, database):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise InvalidImage("The uploaded image could not be decoded.")

        detection = self.detector(frame)
        if len(detection) == 3:
            plate_image, bbox, detection_method = detection
        else:
            plate_image, bbox = detection
            detection_method = (
                "Contour detection"
                if plate_image is not None and bbox is not None
                else None
            )
        if plate_image is None or bbox is None:
            return ProcessingResult(
                "NOT ALLOWED",
                "No plate-like region was found.",
                "",
                None,
                _encode_image(frame),
                None,
            )

        if self._reader is None:
            self._reader = self.ocr_loader()
        ocr_text = self.ocr(
            plate_image,
            self._reader,
            confidence_threshold=self.config.ocr_confidence_threshold,
        ).strip()

        usable = len(ocr_text) >= self.config.min_ocr_length
        match = (
            find_match(
                ocr_text,
                database,
                policy=self.config.matching_policy,
                tolerance=self.config.match_tolerance,
            )
            if usable
            else None
        )
        allowed = match is not None
        if not usable:
            reason = "A plate region was found, but OCR returned no usable number."
        elif allowed:
            reason = "The OCR result exactly matches a fictional demo record."
        else:
            reason = "The OCR result is not registered in the fictional demo data."

        annotated = _annotate(frame, bbox, allowed)
        return ProcessingResult(
            "ALLOWED" if allowed else "NOT ALLOWED",
            reason,
            ocr_text if usable else "",
            match,
            _encode_image(annotated),
            detection_method,
        )

    def review(self, image_path):
        """Detect and OCR an upload without making or storing a decision."""
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise InvalidImage("The uploaded image could not be decoded.")

        plate_image, bbox, detection_method = self.detector(frame)
        if plate_image is None or bbox is None:
            return ImageReview(
                _encode_image(frame),
                None,
                "",
                None,
                "No plate-like region was found. Enter the plate manually.",
            )

        if self._reader is None:
            self._reader = self.ocr_loader()
        ocr_text = self.ocr(
            plate_image,
            self._reader,
            confidence_threshold=self.config.ocr_confidence_threshold,
        ).strip()
        error = None
        if len(ocr_text) < self.config.min_ocr_length:
            ocr_text = ""
            error = "OCR returned no usable number. Enter the plate manually."
        return ImageReview(
            _encode_image(frame),
            _encode_image(plate_image),
            ocr_text,
            detection_method,
            error,
        )


def save_validated_upload(file_storage, directory):
    """Validate image bytes with Pillow and save under a random temporary name."""
    extension = Path(file_storage.filename or "").suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png"}:
        raise InvalidImage("Please upload a JPG, JPEG, or PNG file.")

    payload = file_storage.read()
    try:
        with Image.open(BytesIO(payload)) as image:
            image.verify()
        with Image.open(BytesIO(payload)) as image:
            detected_format = image.format
            image = ImageOps.exif_transpose(image).convert("RGB")
            if detected_format not in ALLOWED_FORMATS:
                raise InvalidImage("The image format is not supported.")
            suffix = ALLOWED_FORMATS[detected_format]
            path = Path(directory) / f"{uuid.uuid4().hex}{suffix}"
            image.save(path, format=detected_format)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImage(
            "The uploaded file is corrupted or is not an image."
        ) from exc
    return path


def remove_temporary_file(path):
    """Remove one known temporary upload, tolerating an already-removed file."""
    if path is not None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _annotate(frame, bbox, allowed):
    output = frame.copy()
    x, y, width, height = bbox
    colour = (42, 168, 77) if allowed else (48, 67, 220)
    cv2.rectangle(output, (x, y), (x + width, y + height), colour, 3)
    return output


def _encode_image(frame):
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise InvalidImage("The result image could not be prepared.")
    encoded = base64.b64encode(np.asarray(buffer).tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
