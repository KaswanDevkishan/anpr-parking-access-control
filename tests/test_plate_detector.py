import cv2
import numpy as np

from plate_detector import (
    LIVE_MAX_AREA_RATIO,
    detect_plate,
    detect_plate_for_upload,
)


def outlined_image(width, height, rectangle=None):
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    if rectangle:
        cv2.rectangle(image, rectangle[0], rectangle[1], (0, 0, 0), 4)
    return image


def test_upload_detects_smaller_plate_with_contours():
    frame = outlined_image(400, 240, ((120, 90), (280, 170)))

    crop, bbox, method = detect_plate_for_upload(frame)

    assert crop is not None
    assert bbox is not None
    assert method == "Contour detection"


def test_upload_accepts_close_up_plate_occupying_most_of_image():
    frame = outlined_image(300, 160, ((10, 10), (290, 150)))

    live_crop, live_bbox = detect_plate(frame)
    crop, bbox, method = detect_plate_for_upload(frame)

    assert live_crop is None
    assert live_bbox is None
    assert crop is not None
    assert bbox is not None
    assert method == "Contour detection"


def test_plausible_close_up_without_contours_uses_full_image_fallback():
    frame = outlined_image(300, 150)

    crop, bbox, method = detect_plate_for_upload(frame)

    assert crop is frame
    assert bbox == (0, 0, 300, 150)
    assert method == "Close-up fallback"


def test_portrait_image_does_not_use_close_up_fallback():
    frame = outlined_image(120, 240)

    assert detect_plate_for_upload(frame) == (None, None, None)


def test_square_image_does_not_use_close_up_fallback():
    frame = outlined_image(200, 200)

    assert detect_plate_for_upload(frame) == (None, None, None)


def test_live_detector_defaults_are_unchanged():
    frame = outlined_image(300, 160, ((10, 10), (290, 150)))

    assert LIVE_MAX_AREA_RATIO == 0.35
    assert detect_plate(frame) == (None, None)
