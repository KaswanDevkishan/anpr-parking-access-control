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
            if 1.5 <= aspect <= 5.5 and 3_000 <= area <= frame_area * 0.35:
                plate_roi = frame[y : y + h, x : x + w]
                return plate_roi, (x, y, w, h)

    return None, None
