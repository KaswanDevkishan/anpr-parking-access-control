"""
test_with_image.py
─────────────────────────────────────────────────────────────
Test plate detection + OCR on a single image file.
Useful when you don't have a car handy — just print a plate number
on paper, photograph it, and test with that.

Usage:
    python test_with_image.py path/to/image.jpg
    python test_with_image.py test_plate.png

The script will:
  1. Show the original image with the detected plate outlined in green
  2. Show the cropped plate region
  3. Print what the OCR read and what the matcher found

Press any key to close the windows.
"""

import sys

import cv2

from config import load_config
from matcher import find_match, load_database
from ocr_engine import load_ocr, read_plate
from plate_detector import detect_plate


def test_image(image_path):
    config = load_config()
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌  Cannot read image: '{image_path}'")
        print("   Make sure the file exists and is a .jpg or .png")
        return

    h, w = frame.shape[:2]
    print(f"Image loaded: {w}×{h} pixels")

    # Detect plate
    print("Detecting plate …")
    plate_img, bbox = detect_plate(frame)

    if plate_img is None:
        print("❌  No plate found in this image.")
        print("   Tips:")
        print("   - Make sure the plate is well-lit and mostly facing the camera")
        print("   - Try a closer crop of the vehicle")
        cv2.imshow("Input (no plate found)", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    x, y, w_box, h_box = bbox
    print(f"✅  Plate detected at ({x}, {y}), size {w_box}×{h_box}")

    # OCR
    print("Running OCR …")
    reader = load_ocr()
    ocr_text = read_plate(
        plate_img,
        reader,
        confidence_threshold=config.ocr_confidence_threshold,
    )
    print(f"✅  OCR result: '{ocr_text}'")

    # Database match
    db = load_database(config.database_path)
    owner = find_match(
        ocr_text,
        db,
        policy=config.matching_policy,
        tolerance=config.match_tolerance,
    )

    if owner:
        print(f"✅  MATCH FOUND: {owner['name']}  ({owner['id']},  {owner['type']})")
        colour = (0, 200, 0)
        label = f"ALLOWED: {owner['name']}"
    else:
        print(f"❌  No match for '{ocr_text}' — NOT ALLOWED")
        colour = (0, 0, 220)
        label = f"NOT ALLOWED: [{ocr_text}]"

    # Draw result on image
    annotated = frame.copy()
    cv2.rectangle(annotated, (x, y), (x + w_box, y + h_box), colour, 3)
    cv2.putText(
        annotated,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        colour,
        2,
    )

    print("\nShowing result windows — press any key to close.")
    try:
        cv2.imshow("Detection Result", annotated)
        cv2.imshow("Cropped Plate", plate_img)
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:  python test_with_image.py <path_to_image>")
        print("Example: python test_with_image.py test_plate.jpg")
    else:
        test_image(sys.argv[1])
