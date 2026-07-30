"""
test_camera.py
─────────────────────────────────────────────────────────────
Run this FIRST to confirm your camera works before anything else.

Usage:
    python test_camera.py          # tries camera index 0 (built-in Mac cam)
    python test_camera.py 1        # tries camera index 1 (USB cam)

Press 'q' in the window to quit.
"""

import sys

import cv2

# Camera index: 0 = built-in Mac FaceTime cam, 1 = first USB cam
index = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def main():
    print(f"Trying camera index {index}...")
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            print(f"\n❌  Could not open camera {index}.")
            print("   Try:  python test_camera.py 1")
            print(
                "   Or check: System Settings → Privacy & Security → "
                "Camera → allow Terminal"
            )
            return 1

        print(f"✅  Camera {index} opened successfully!")
        print("   A window should appear. Press 'q' to close it.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌  Camera stopped sending frames.")
                break

            h, w = frame.shape[:2]
            label = f"Camera {index}  |  {w}x{h}  |  press Q to quit"
            cv2.putText(
                frame,
                label,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 0),
                2,
            )
            cv2.imshow("Camera Test", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("Camera test complete ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
