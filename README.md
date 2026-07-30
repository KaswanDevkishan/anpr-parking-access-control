# ANPR Access-Decision Prototype

A compact automatic number-plate recognition (ANPR) prototype that demonstrates
how a camera frame can be processed into an access decision. It was built to
explore practical computer vision, OCR, privacy-aware data handling, and
explainable authorization logic on ordinary computers and Raspberry Pi-class
hardware.

> **Prototype scope:** this software displays an access recommendation only. It
> is not production-ready, does not control a gate, and must not be connected to
> barriers, GPIO, locks, or other physical access equipment.

## Architecture and processing flow

The project deliberately keeps each processing concern small and independent:

1. `main.py` captures frames and owns the display loop.
2. `plate_detector.py` uses OpenCV contours to identify a plate-like rectangle.
3. `ocr_engine.py` preprocesses the crop and reads text with EasyOCR.
4. `matcher.py` normalises the result, validates the CSV database, and makes an
   authorization match.
5. `config.py` provides validated environment configuration and
   repository-relative paths.
6. The UI overlays an `ALLOWED` or `NOT ALLOWED` decision. No physical action is
   performed.

## Technology stack

- Python 3.9+
- OpenCV for camera input, image processing, contour detection, and overlays
- EasyOCR for optical character recognition
- NumPy for OpenCV/EasyOCR numerical operations
- python-Levenshtein for optional experimental fuzzy matching
- pytest and Ruff for automated checks

## Features

- Live camera processing with configurable camera index
- Static-image workflow for repeatable demonstrations
- Lightweight contour-based plate-region detection
- Japanese and English EasyOCR reader configuration
- Exact authorization matching by default
- Explicit, opt-in fuzzy matching that rejects ambiguous closest-match ties
- Validated CSV input with duplicate-plate detection
- Repository-relative, environment-overridable configuration
- Screenshot capture using `S` or `s`

## Setup

Python 3.9 or newer is recommended. EasyOCR may download language model files on
first use, so the first run can take longer and requires network access.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The included `data/vehicles.example.csv` contains fictional demonstration data.
For private local records, create `data/vehicles.csv`; it is ignored by Git.
Use the same columns:

```csv
plate_number,name,id,type
YOURPLATE,Local Name,LOCAL-ID,staff
```

Configuration is read directly from environment variables. `.env.example`
documents every option, but `.env` files are not automatically loaded:

```bash
export ANPR_DATABASE_PATH=data/vehicles.csv
export ANPR_MATCHING_POLICY=exact
```

Relative database and screenshot paths resolve from the repository directory,
not the terminal's working directory.

## Live-camera usage

```bash
python3 main.py
python3 main.py 1  # command-line camera index overrides the environment
```

Press `Q` or `q` to quit and `S` or `s` to save a screenshot. Camera index can
also be set with `ANPR_CAMERA_INDEX`. Use `python3 test_camera.py 0` for a simple
camera check.

## Static-image usage

Use only fictional or properly consented test images, and do not commit them:

```bash
python3 test_with_image.py /path/to/local/test-image.jpg
```

The script runs the same detector, OCR confidence threshold, database, and
matching policy as the live workflow, then displays the annotated result.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANPR_CAMERA_INDEX` | `0` | OpenCV camera device index |
| `ANPR_DATABASE_PATH` | `data/vehicles.example.csv` | Vehicle CSV path |
| `ANPR_OCR_EVERY_N` | `30` | Frames between OCR attempts |
| `ANPR_OCR_CONFIDENCE_THRESHOLD` | `0.5` | Minimum OCR segment confidence |
| `ANPR_MATCHING_POLICY` | `exact` | `exact` or experimental `fuzzy` |
| `ANPR_MATCH_TOLERANCE` | `1` | Fuzzy edit-distance limit |
| `ANPR_SCREENSHOT_DIR` | `screenshots` | Local screenshot output |
| `ANPR_COOLDOWN_SECONDS` | `3` | Repeat log cooldown |
| `ANPR_MIN_OCR_LENGTH` | `3` | Minimum accepted OCR text length |

Fuzzy matching is experimental. It can increase false-positive risk, and a tie
between equally close plates is always rejected. Exact matching should remain
the authorization policy.

## Privacy

Licence plates and identity records are personal or sensitive data in many
contexts. This repository contains fictional records only. Never commit real
plates, names, student or staff IDs, screenshots, OCR model downloads, secrets,
or generated user data. Keep private CSV files at `data/vehicles.csv` or another
ignored local path, establish consent and retention rules, and secure any
real-world deployment data outside this repository.

## Limitations

- The contour heuristic is sensitive to lighting, angle, blur, plate design,
  and background geometry.
- OCR accuracy is not guaranteed and the current pipeline has no calibrated
  confidence for the final access decision.
- The example database is an in-memory CSV, without authentication, auditing,
  encryption, concurrency controls, or lifecycle management.
- The interface requires a GUI and the live path depends on camera/OpenCV
  support on the host.
- The system has not undergone security, privacy, accessibility, safety, load,
  or field validation.

## Access decision, not gate control

This project ends at an on-screen recommendation. An `ALLOWED` label is a
software demonstration, not proof of identity or permission. A real access
system requires defense-in-depth, human-safe failure modes, audited hardware,
manual override, legal/privacy review, monitoring, and independent security and
safety testing. Those controls are intentionally outside this prototype.

## Future improvements

- Replace contour heuristics with a separately evaluated plate detector
- Add representative, consented test fixtures and accuracy benchmarks
- Return structured OCR confidence and require multi-frame consensus
- Add schema versioning and a secure database abstraction
- Improve observability without retaining raw personal imagery
- Package a headless Raspberry Pi runtime while preserving a local demo mode
- Conduct threat modeling, bias evaluation, privacy review, and field testing

## Screenshots and demo

<!-- Add only fictional/synthetic imagery. Never commit real plate captures. -->

**Screenshot placeholder:** annotated synthetic plate detection image.

**Demo-video placeholder:** short synthetic live-camera walkthrough.

## Development checks

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m compileall .
python3 -m pytest -v
python3 -m ruff check .
```

## Resume-ready description

> Built a modular Python ANPR access-decision prototype using OpenCV and
> EasyOCR, with live-camera and static-image workflows, validated CSV matching,
> privacy-safe configuration, ambiguity-aware authorization rules, and
> camera-independent pytest coverage.

## License

Released under the [MIT License](LICENSE).

