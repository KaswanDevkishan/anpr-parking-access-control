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
4. `matcher.py` normalises results and applies exact authorization matching.
5. `config.py` provides validated environment configuration and
   repository-relative paths.
6. The Raspberry Pi CLI reads CSV; the Flask application uses SQLite through
   `anpr_web/database.py`.
7. The UI renders an annotated `ALLOWED` or `NOT ALLOWED` recommendation. No
   physical action is performed.

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
- Server-rendered Flask portfolio demo with private-by-default temporary uploads
- Public searchable fictional vehicle registry backed by SQLite
- Session-authenticated, CSRF-protected single-admin vehicle management
- Manual and image-assisted registration with explicit confirmation
- Persistent entry/exit visits with searchable, paginated access history
- Admin-only browser-camera registration with multi-sample OCR confirmation
- Accessible light/dark themes with saved browser preference
- Headless production WSGI entry point and health check

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

## Web application

The Flask interface is a separate, headless access-decision portfolio layer. It
reuses detection, OCR, normalization, and exact matching without calling
`VideoCapture`, `cv2.imshow`, or the Raspberry Pi display loop. Its vehicle
records are stored in SQLite; the live-camera CLI continues to use CSV.

Run locally:

```bash
export ANPR_SQLITE_PATH=data/anpr_web.sqlite3
export ANPR_SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
export ANPR_ADMIN_USERNAME=your-local-admin-name
export ANPR_ADMIN_PASSWORD_HASH="$(python3 -c 'from werkzeug.security import generate_password_hash; import getpass; print(generate_password_hash(getpass.getpass()))')"
export FLASK_APP=wsgi:app
export FLASK_DEBUG=1
flask run
```

The SQLite schema is migrated idempotently at startup. Existing vehicles and
visits are preserved; missing tables, columns, and indexes are added without
recreating the database. If the `vehicles` table is empty, fictional rows are
seeded once from `data/vehicles.example.csv`; later startups never overwrite
existing records.

Every successful exact active-vehicle match applies the configured checkpoint
action. Entry creates one open visit; another entry returns **Already inside**.
Exit closes that open visit; an exit without one returns **No active entry
found**. Inactive, deleted, unreadable, and unregistered scans never alter visit
history. Completed visits are retained rather than overwritten so the history
remains auditable. Timestamps come from the server, are stored in UTC, and are
displayed in `ANPR_TIMEZONE` (Asia/Tokyo by default).

`ANPR_CHECKPOINT_MODE` supports `entry`, `exit`, and `selectable`. Selectable
mode requires the operator to choose Entry or Exit before analysing an image;
the prototype never infers direction from a still image. Access History supports
plate/display-name search, status and local-date filters, newest-first
pagination, durations, and admin-only aggregate totals.

Open `/admin/login` and use the configured admin credentials. The dashboard
supports manual registration, image-assisted registration, activation,
deactivation, browser-camera registration, and explicitly confirmed deletion.
All registration methods use the same normalized server-side validation and
persistence path. Image-assisted registration
only pre-fills a review form: an administrator must correct and confirm the
record before it is saved. No public registration or additional roles exist.

Register with Camera uses the client browser's
`navigator.mediaDevices.getUserMedia()` API; the server never attempts to open
the client camera. It uploads low-frequency compressed still samples to an
admin/CSRF-protected endpoint, requires repeated stable OCR results for automatic
review, times out conservatively, and retains no image files. OCR alone never
creates a vehicle: the admin must review/correct the plate and explicitly
confirm all fields. Camera permission denial or unsupported browsers can use
the existing upload and manual alternatives. Camera access normally requires
HTTPS or localhost.

The navigation theme toggle respects `prefers-color-scheme` on first visit,
stores an explicit choice in `localStorage`, and changes without reloading.

Vehicle records may include an optional administrator-only email address and an
explicit consent checkbox for exit summaries. Manual, image-assisted, and
browser-camera registration share one validation and persistence service; OCR
never invents an address. On a successful exit, SQLite commits first, then one
idempotent delivery record is created. SMTP failure cannot reverse the exit or
change the access decision. Failed deliveries can be retried by an authenticated
administrator using a CSRF-protected POST action; sent deliveries cannot be
resent. The local `dry-run` backend avoids network access and retains messages
in memory only.

The admin dashboard reports open visits for current vehicles. A vehicle with an
open visit must exit before an administrator can permanently delete its registry
row. Completed visit and masked notification-delivery history remains available
through immutable vehicle snapshots. When `ANPR_PARKING_CAPACITY` is configured,
the dashboard also shows available spaces and an occupancy percentage; capacity
never blocks access. Authenticated administrators can export the active history
filters directly to an in-memory CSV. Formula-like cells are neutralized and
email addresses, IDs, and delivery errors are omitted. The paginated audit log
records selected vehicle and retry actions without raw email addresses or
secrets.

Uploads are limited to 8 MB, verified with Pillow, assigned random temporary
names, and deleted after each request. Original and cropped image-registration
images are not retained by default. Use only synthetic or properly consented
images.

Render is configured by `render.yaml`, including a persistent disk mounted at
`/var/data`. SQLite changes disappear across restarts or redeployments without
persistent storage; do not scale this SQLite deployment to multiple independent
instances. Configure `ANPR_ADMIN_USERNAME` and
`ANPR_ADMIN_PASSWORD_HASH` as secret environment values in Render. The
production start command is:

```bash
gunicorn wsgi:app
```

The service health check is available at `/health`. EasyOCR remains lazy-loaded,
although the first analysis request may be slow while the OCR model initializes.
Keep `ANPR_MATCHING_POLICY=exact` for the public demo.

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
| `ANPR_WEB_TEMP_DIR` | OS temporary directory | Ephemeral web upload directory |
| `ANPR_SQLITE_PATH` | `data/anpr_web.sqlite3` | Flask vehicle database path |
| `ANPR_TIMEZONE` | `Asia/Tokyo` | IANA timezone used for display and local-date reporting |
| `ANPR_CHECKPOINT_MODE` | `entry` | Web checkpoint action: `entry`, `exit`, or `selectable` |
| `ANPR_SECRET_KEY` | Random per process | Persistent Flask session signing secret |
| `ANPR_ADMIN_USERNAME` | Empty | Single admin login name |
| `ANPR_ADMIN_PASSWORD_HASH` | Empty | Werkzeug password hash; never plaintext |
| `ANPR_APPLICATION_NAME` | `Example Campus Parking` | Fictional name used in summaries |
| `ANPR_PARKING_CAPACITY` | Empty | Optional positive capacity for occupancy reporting |
| `ANPR_EMAIL_ENABLED` | `false` | Enables configured exit-summary delivery |
| `ANPR_EMAIL_BACKEND` | `smtp` | `smtp` or local/test `dry-run` backend |
| `ANPR_SMTP_HOST` | Empty | SMTP server hostname |
| `ANPR_SMTP_PORT` | `587` | Positive SMTP port |
| `ANPR_SMTP_USERNAME` | Empty | Optional SMTP username; configure as a secret |
| `ANPR_SMTP_PASSWORD` | Empty | Optional SMTP password; configure as a secret |
| `ANPR_SMTP_USE_TLS` | `true` | Upgrade SMTP connection with STARTTLS |
| `ANPR_EMAIL_FROM` | Empty | Sender address required when email is enabled |
| `ANPR_EMAIL_FROM_NAME` | `Example Campus Parking` | Human-readable sender name |
| `ANPR_EMAIL_TIMEOUT_SECONDS` | `5` | Short positive network timeout |

Fuzzy matching is experimental. It can increase false-positive risk, and a tie
between equally close plates is always rejected. Exact matching should remain
the authorization policy.

## Privacy

Licence plates and identity records are personal or sensitive data in many
contexts. This repository contains fictional records only. Never commit real
plates, names, student or staff IDs, SQLite databases, screenshots, OCR model
downloads, secrets, credentials, or generated user data. Keep private CSV and
SQLite files in ignored locations, establish consent and retention rules, and
secure any real-world deployment data outside this repository. The public
registry deliberately exposes only plate number, fictional display owner,
category, and active status.
Email notification consent is optional. Delivery history stores only a masked
destination, status, timestamps, and a generic error code; message bodies and
SMTP responses are not stored. Provider configuration belongs in deployment
secrets, never source files. See `/privacy` for the in-application summary.

On Render, keep `ANPR_EMAIL_ENABLED=false` until all SMTP secret variables have
been configured. The persistent `/var/data` disk is required for vehicle,
visit, delivery, and audit records; ephemeral or multi-instance SQLite
deployments can lose data or contend on writes.

## Limitations

- The contour heuristic is sensitive to lighting, angle, blur, plate design,
  and background geometry.
- OCR accuracy is not guaranteed and the current pipeline has no calibrated
  confidence for the final access decision.
- SQLite is suitable for this portfolio workflow but does not provide auditing,
  application-level encryption or multi-instance write coordination. The
  application audit log is selective and is not a tamper-evident security log.
- Email delivery is synchronous with a short timeout; a production system would
  use a durable queue, provider event reconciliation, verified ownership, and
  formal retention/consent processes.
- Browser-camera consensus reduces accidental captures but does not establish
  OCR accuracy, vehicle identity, or direction of travel.
- Login throttling is process-local and resets on restart; a distributed
  deployment would need a shared rate-limit store.
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

**Live demo URL:** `https://YOUR-RENDER-SERVICE.onrender.com` *(placeholder)*

**Homepage screenshot:** `docs/screenshots/homepage-placeholder.png` *(placeholder)*

**Result screenshot:** `docs/screenshots/result-placeholder.png` *(placeholder)*

**Demo video:** short synthetic live-camera walkthrough *(placeholder)*

## Development checks

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m compileall .
python3 -m pytest -v
python3 -m ruff check .
python3 -m ruff format --check .
git diff --check
gunicorn --check-config wsgi:app
```

## Description

> Built a modular Python ANPR access-decision prototype using OpenCV and
> EasyOCR, with an unchanged Raspberry Pi CSV workflow and a Flask/SQLite
> portfolio application featuring exact active-record authorization,
> privacy-aware uploads, admin authentication, CSRF protection, and explicit
> vehicle lifecycle management.

## License

Released under the [MIT License](LICENSE).
