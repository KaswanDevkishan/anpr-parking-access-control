import cv2

from ocr_engine import load_ocr, read_plate
from plate_detector import detect_plate


def main():
    cap = cv2.VideoCapture(0)
    reader = load_ocr()

    print("SPACE = capture and run OCR | Q = quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()

            # Show ALL rectangles the detector considers
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 200)
            cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:15]:
                approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    cv2.rectangle(display, (x, y), (x + w, y + h), (200, 200, 0), 1)

            # Green box = plate the system actually accepts
            plate_img, bbox = detect_plate(frame)
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 220, 0), 3)
                cv2.putText(
                    display,
                    "PLATE FOUND",
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 220, 0),
                    2,
                )

            cv2.putText(
                display,
                "Yellow=candidates  Green=accepted plate  SPACE=OCR  Q=quit",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            cv2.imshow("Debug", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                print("\n--- OCR on whole frame ---")
                result = read_plate(frame, reader)
                print(f"OCR says: '{result}'")
                print("--- OCR on plate region ---")
                if plate_img is not None:
                    result2 = read_plate(plate_img, reader)
                    print(f"Plate OCR says: '{result2}'")
                else:
                    print("No plate region detected")
            elif key in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
