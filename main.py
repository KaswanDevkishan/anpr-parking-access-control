"""
main.py
─────────────────────────────────────────────────────────────
University Parking Access Control System — main entry point.

Run with:
    python main.py          # camera index 0 (built-in Mac camera)
    python main.py 1        # camera index 1 (USB camera)

Controls:
    Q   →  quit
    S   →  save a screenshot of the current frame
"""

import sys
import time
from datetime import datetime

import cv2

from config import load_config
from matcher import find_match, load_database
from ocr_engine import load_ocr, read_plate
from plate_detector import detect_plate

WINDOW_TITLE = "University Parking – Access-Decision Prototype"

COLOUR_ALLOWED = (0, 200, 0)  # green  (BGR)
COLOUR_NOT_ALLOWED = (0, 0, 220)  # red    (BGR)
COLOUR_INFO = (200, 200, 200)  # light grey for overlays


# ── Helpers ──────────────────────────────────────────────────────────────────


def draw_result(frame, ocr_text, owner, bbox):
    """Draw coloured bounding box and label on the frame."""
    x, y, w, h = bbox

    if owner:
        colour = COLOUR_ALLOWED
        label = f"ALLOWED    {owner['name']}  ({owner['id']})"
    else:
        colour = COLOUR_NOT_ALLOWED
        label = f"NOT ALLOWED    [{ocr_text}]"

    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.72
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    text_x = x
    text_y = max(y - 8, th + baseline + 4)

    cv2.rectangle(
        frame,
        (text_x - 2, text_y - th - baseline - 4),
        (text_x + tw + 4, text_y + baseline),
        colour,
        -1,
    )
    cv2.putText(
        frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness
    )


def draw_status_bar(frame):
    cv2.putText(
        frame,
        "ANPR Access Decision  |  Q = quit  |  S = screenshot",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COLOUR_INFO,
        1,
    )


def save_screenshot(frame, directory):
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = directory / f"frame_{timestamp}.png"
    cv2.imwrite(str(filename), frame)
    print(f"   Screenshot saved: {filename}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    config = load_config()
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else config.camera_index

    print("=" * 55)
    print("  University Parking – ANPR Access-Decision Prototype")
    print("=" * 55)

    print("\n[1/3] Loading OCR engine …")
    reader = load_ocr()

    print("\n[2/3] Loading vehicle database …")
    db = load_database(config.database_path)
    if not db:
        print("Stopping — fix the database file and try again.")
        return

    print(f"\n[3/3] Opening camera (index {camera_index}) …")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"\n  Cannot open camera {camera_index}.")
        print("   Try:  python main.py 1")
        cap.release()
        return

    # ── Force lower resolution for smoother performance ────────────────────
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   Camera ready  ({w}×{h})\n")
    print("System running. Press 'Q' in the video window to quit.")
    print("-" * 55)

    frame_count = 0
    last_result = None
    last_plate_text = ""
    last_match_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera stopped — lost feed.")
                break

            frame_count += 1
            display = frame.copy()

            # ── Run detection + OCR every N frames ──────────────────────────
            if frame_count % config.ocr_every_n == 0:
                plate_img, bbox = detect_plate(frame)

                if plate_img is not None:
                    ocr_text = read_plate(
                        plate_img,
                        reader,
                        confidence_threshold=config.ocr_confidence_threshold,
                    )

                    if len(ocr_text) >= config.min_ocr_length:
                        owner = find_match(
                            ocr_text,
                            db,
                            policy=config.matching_policy,
                            tolerance=config.match_tolerance,
                        )
                        last_result = (ocr_text, owner, bbox)

                        now = time.time()
                        elapsed = now - last_match_time
                        if (
                            ocr_text != last_plate_text
                            or elapsed > config.cooldown_seconds
                        ):
                            status = "ALLOWED" if owner else "NOT ALLOWED"
                            name = owner["name"] if owner else "—"
                            distance = f"  dist={owner['distance']}" if owner else ""
                            print(
                                f"  OCR: '{ocr_text}'{distance}  →  {status}  ({name})"
                            )
                            last_plate_text = ocr_text
                            last_match_time = now
                    else:
                        last_result = None
                        last_plate_text = ""
                        last_match_time = 0
                else:
                    last_result = None
                    last_plate_text = ""
                    last_match_time = 0

            # ── Draw cached result on every frame ────────────────────────────
            if last_result:
                draw_result(display, *last_result)

            draw_status_bar(display)
            cv2.imshow(WINDOW_TITLE, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("\nQ pressed — shutting down.")
                break
            if key in (ord("s"), ord("S")):
                save_screenshot(display, config.screenshot_directory)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("System stopped. Goodbye!")


if __name__ == "__main__":
    main()
