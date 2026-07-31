"""
plate_detector.py
─────────────────────────────────────────────────────────────
Finds the licence plate region inside a video frame.

How it works:
  1. Convert frame to grayscale and blur it
  2. Find edges using Canny edge detection
  3. Find all rectangular contours (shapes with 4 corners)
  4. Pick the one with the right aspect ratio for a licence plate

Returns the cropped plate image + its coordinates (x, y, w, h).
"""

import cv2

LIVE_MAX_AREA_RATIO = 0.35
UPLOAD_MAX_AREA_RATIO = 0.9
CLOSE_UP_MIN_ASPECT_RATIO = 1.8
CLOSE_UP_MAX_ASPECT_RATIO = 2.4
UPLOAD_HORIZONTAL_PADDING_RATIO = 0.10


def _preprocess(frame):
    """Prepare frame for contour detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 200)

    # Dilate edges slightly to close small gaps in plate border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    return edges


def detect_plate(frame):
    """
    Locate a licence plate rectangle in a single video frame.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR image from cv2.VideoCapture.read()

    Returns
    -------
    (plate_img, (x, y, w, h))  –  if a plate-like region is found
    (None, None)                –  if nothing suitable is found
    """
    return _detect_plate_with_max_area(frame, LIVE_MAX_AREA_RATIO)


def detect_plate_for_upload(frame):
    """
    Locate a plate in an uploaded image without changing live-camera defaults.

    The live detector is tried first. A second contour pass permits a plate to
    occupy more of an upload. If neither pass finds a contour, a landscape image
    with a plausible plate aspect ratio is treated conservatively as a close-up
    crop and passed whole to OCR.

    Returns
    -------
    (plate_img, (x, y, w, h), method)  –  if a plate-like region is found
    (None, None, None)                  –  if nothing suitable is found
    """
    plate_image, bbox = detect_plate(frame)
    if plate_image is not None and bbox is not None:
        plate_image, bbox = _pad_upload_crop(frame, bbox)
        return plate_image, bbox, "Contour detection"

    plate_image, bbox = _detect_plate_with_max_area(frame, UPLOAD_MAX_AREA_RATIO)
    if plate_image is not None and bbox is not None:
        plate_image, bbox = _pad_upload_crop(frame, bbox)
        return plate_image, bbox, "Contour detection"

    height, width = frame.shape[:2]
    aspect_ratio = width / float(height)
    if CLOSE_UP_MIN_ASPECT_RATIO <= aspect_ratio <= CLOSE_UP_MAX_ASPECT_RATIO:
        return frame, (0, 0, width, height), "Close-up fallback"

    return None, None, None


def _pad_upload_crop(frame, bbox):
    """Keep horizontal context around an upload contour for lower-row OCR."""
    x, y, width, height = bbox
    frame_width = frame.shape[1]
    padding = max(1, round(width * UPLOAD_HORIZONTAL_PADDING_RATIO))
    left = max(0, x - padding)
    right = min(frame_width, x + width + padding)
    return frame[y : y + height, left:right], (left, y, right - left, height)


def _detect_plate_with_max_area(frame, max_area_ratio):
    """Run contour detection with an explicit maximum candidate area."""
    edges = _preprocess(frame)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Work with the 30 largest contours only (plate will be prominent)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    frame_area = frame.shape[0] * frame.shape[1]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Licence plates are rectangles → 4 corners
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / float(h)
            area = w * h

            # Japanese plates are ~330×165 mm (2:1 ratio).
            # Accept 1.5–5.5 to handle slight angles and partial views.
            # Area bounds: not too small (noise) and not the whole frame.
            if 1.5 <= aspect <= 5.5 and 3_000 <= area <= frame_area * max_area_ratio:
                plate_roi = frame[y : y + h, x : x + w]
                return plate_roi, (x, y, w, h)

    return None, None
