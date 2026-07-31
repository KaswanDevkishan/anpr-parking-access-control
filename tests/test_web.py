import re
import urllib.error
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from werkzeug.security import generate_password_hash

from anpr_web import create_app
from anpr_web.database import VehicleStore
from anpr_web.mail import HTTPAPIBackend
from anpr_web.processing import ImageReview, ProcessingResult, WebProcessor
from config import load_config

CSRF_PATTERN = re.compile(rb'name="csrf_token" value="([^"]+)"')


def image_bytes(image_format="PNG"):
    stream = BytesIO()
    Image.new("RGB", (100, 60), "white").save(stream, format=image_format)
    return stream.getvalue()


@pytest.fixture
def seed_path(tmp_path):
    path = tmp_path / "vehicles.csv"
    path.write_text(
        "plate_number,name,id,type\n"
        "1001,Alex Example,DEMO-STUDENT-001,student\n"
        "2002,Casey Sample,DEMO-STAFF-001,staff\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def runtime_config(seed_path):
    return load_config({"ANPR_DATABASE_PATH": str(seed_path)})


@pytest.fixture
def app_factory(seed_path, tmp_path):
    counter = 0

    def factory(processor=None, **overrides):
        nonlocal counter
        counter += 1
        config = {
            "TESTING": True,
            "SECRET_KEY": "test-only-session-key",
            "SQLITE_PATH": str(tmp_path / f"web-{counter}.sqlite3"),
            "SEED_CSV_PATH": str(seed_path),
            "UPLOAD_TEMP_DIR": str(tmp_path / f"uploads-{counter}"),
            "ADMIN_USERNAME": "portfolio-admin",
            "ADMIN_PASSWORD_HASH": generate_password_hash("test-password"),
            "ADMIN_LOGIN_MAX_ATTEMPTS": 3,
            "ADMIN_LOCKOUT_SECONDS": 60,
            "ANPR_CHECKPOINT_MODE": "entry",
        }
        config.update(overrides)
        return create_app(config, processor=processor)

    return factory


@pytest.fixture
def client(app_factory):
    return app_factory(StubProcessor()).test_client()


class StubProcessor:
    def __init__(self, review=None):
        self.path = None
        self.existed_during_processing = False
        self.review_result = review

    def process(self, path, _database):
        self.path = Path(path)
        self.existed_during_processing = self.path.exists()
        return ProcessingResult(
            "NOT ALLOWED", "Test result", "9999", None, "data:image/jpeg;base64,"
        )

    def review(self, path):
        self.path = Path(path)
        self.existed_during_processing = self.path.exists()
        return self.review_result or ImageReview(
            "data:image/jpeg;base64,original",
            "data:image/jpeg;base64,crop",
            "3003",
            "Contour detection",
            None,
        )


def csrf(client, path="/"):
    response = client.get(path)
    match = CSRF_PATTERN.search(response.data)
    assert match, f"No CSRF token rendered by {path}"
    return match.group(1).decode()


def post(client, path, data=None, **kwargs):
    payload = dict(data or {})
    payload["csrf_token"] = csrf(client, path if path.endswith("/login") else "/")
    return client.post(path, data=payload, **kwargs)


def login(client, password="test-password"):
    return post(
        client,
        "/admin/login",
        {"username": "portfolio-admin", "password": password},
    )


def post_image(client, path="/analyse", payload=None, filename="vehicle.png"):
    payload = image_bytes() if payload is None else payload
    return post(
        client,
        path,
        {"vehicle_image": (BytesIO(payload), filename)},
        content_type="multipart/form-data",
    )


def post_camera_frame(client, payload=None, filename="camera-frame.jpg"):
    payload = image_bytes("JPEG") if payload is None else payload
    return post(
        client,
        "/admin/vehicles/from-camera/frame",
        {"frame": (BytesIO(payload), filename)},
        content_type="multipart/form-data",
    )


def vehicle_form(plate="3003", name="Jordan Fiction", category="visitor"):
    return {
        "plate_number": plate,
        "display_name": name,
        "category": category,
        "is_active": "1",
    }


def test_demo_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Smart vehicle access. Instant decisions." in response.data
    assert b"Use synthetic or consented images." in response.data


def test_architecture_page_loads(client):
    response = client.get("/architecture")
    assert response.status_code == 200
    assert b"Raspberry Pi live camera" in response.data
    assert b"exact normalized match" in response.data


def test_registered_vehicles_page_is_public_and_private_fields_are_hidden(client):
    response = client.get("/registered-vehicles")
    assert response.status_code == 200
    assert b"Alex Example" in response.data
    assert b"DEMO-STUDENT-001" not in response.data
    assert b"Registered Vehicles" in response.data


def test_vehicle_search_and_category_filter_work(client):
    assert b"Alex Example" in client.get("/registered-vehicles?q=1+0+0+1").data
    response = client.get("/registered-vehicles?category=staff")
    assert b"Casey Sample" in response.data
    assert b"Alex Example" not in response.data


def test_inactive_vehicle_is_shown(client):
    store = client.application.extensions["anpr_store"]
    vehicle = next(row for row in store.list_all() if row["plate_number"] == "1001")
    store.set_active(vehicle["id"], False)
    response = client.get("/registered-vehicles")
    assert b"Inactive" in response.data


def test_public_pages_cannot_modify_records(client):
    store = client.application.extensions["anpr_store"]
    vehicle = store.list_all()[0]
    response = client.get(f"/admin/vehicles/{vehicle['id']}/deactivate")
    assert response.status_code == 405
    assert store.get(vehicle["id"])["is_active"] == 1


def test_login_succeeds_and_dashboard_navigation_changes(client):
    response = login(client)
    assert response.status_code == 302
    dashboard = client.get("/admin")
    assert b"Admin Dashboard" in dashboard.data
    assert b"Demo storage is temporary and may reset." in dashboard.data
    assert b"Logout" in dashboard.data


def test_invalid_login_uses_generic_failure(client):
    response = login(client, "wrong-password")
    assert response.status_code == 200
    assert b"Login failed. Please check your credentials" in response.data


def test_protected_dashboard_redirects_unauthenticated_user(client):
    response = client.get("/admin")
    assert response.status_code == 302
    assert "/admin/login" in response.headers["Location"]


def test_logout_works(client):
    login(client)
    response = post(client, "/admin/logout")
    assert response.status_code == 302
    assert client.get("/admin").status_code == 302


def test_login_lockout_works(client):
    for _ in range(3):
        login(client, "wrong-password")
    response = login(client)
    assert response.status_code == 429
    assert b"try again later" in response.data


def test_csrf_rejection_works(client):
    response = client.post(
        "/admin/login",
        data={"username": "portfolio-admin", "password": "test-password"},
    )
    assert response.status_code == 400


def test_add_valid_vehicle(client):
    login(client)
    response = post(client, "/admin/vehicles/add", vehicle_form())
    assert response.status_code == 302
    assert client.application.extensions["anpr_store"].list_public("3003")


def test_duplicate_normalized_plate_is_rejected(client):
    login(client)
    response = post(
        client,
        "/admin/vehicles/add",
        vehicle_form("1 0 0 1"),
    )
    assert b"normalized plate number is already registered" in response.data


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"category": "owner"}, b"Choose a valid vehicle category"),
        ({"plate_number": ""}, b"Plate number is required"),
        ({"display_name": ""}, b"Display owner name is required"),
    ],
)
def test_invalid_manual_vehicle_fields_are_rejected(client, changes, message):
    login(client)
    data = vehicle_form()
    data.update(changes)
    response = post(client, "/admin/vehicles/add", data)
    assert response.status_code == 200
    assert message in response.data


def test_deactivate_and_reactivate_vehicle(client):
    login(client)
    store = client.application.extensions["anpr_store"]
    vehicle = store.list_all()[0]
    post(client, f"/admin/vehicles/{vehicle['id']}/deactivate")
    assert store.get(vehicle["id"])["is_active"] == 0
    post(client, f"/admin/vehicles/{vehicle['id']}/reactivate")
    assert store.get(vehicle["id"])["is_active"] == 1


def test_delete_requires_post_and_exact_confirmation(client):
    anonymous = client.application.test_client()
    assert anonymous.get("/admin/vehicles/1/delete").status_code == 302
    assert anonymous.post("/admin/vehicles/1/delete").status_code == 400
    login(client)
    store = client.application.extensions["anpr_store"]
    vehicle = store.list_all()[0]
    assert client.post(f"/admin/vehicles/{vehicle['id']}/delete").status_code == 400
    confirm = client.get(f"/admin/vehicles/{vehicle['id']}/delete")
    assert b"cannot be undone" in confirm.data
    assert store.get(vehicle["id"]) is not None
    rejected = post(
        client,
        f"/admin/vehicles/{vehicle['id']}/delete",
        {"confirmation": "WRONG"},
    )
    assert rejected.status_code == 400
    assert store.get(vehicle["id"]) is not None
    accepted = post(
        client,
        f"/admin/vehicles/{vehicle['id']}/delete",
        {"confirmation": vehicle["plate_number"]},
    )
    assert accepted.status_code == 302
    assert store.get(vehicle["id"]) is None
    assert not store.list_public(vehicle["plate_number"])
    replacement_id = store.create(
        vehicle["plate_number"], "Replacement Fiction", "visitor"
    )
    assert replacement_id != vehicle["id"]
    audit_rows, _ = store.audit_log()
    deletion = next(row for row in audit_rows if row["action"] == "vehicle_deleted")
    assert vehicle["plate_number"] not in deletion["target_reference"]


def test_delete_with_open_visit_is_blocked(client):
    login(client)
    store = client.application.extensions["anpr_store"]
    vehicle = store.list_all()[0]
    store.apply_checkpoint(vehicle["plate_number"], "entry")

    response = post(
        client,
        f"/admin/vehicles/{vehicle['id']}/delete",
        {"confirmation": vehicle["plate_number"]},
    )

    assert response.status_code == 409
    assert (
        b"This vehicle is currently inside. Record its exit before deleting it."
        in response.data
    )
    assert store.get(vehicle["id"]) is not None


def test_valid_image_reaches_editable_ocr_review_and_cleans_temp(app_factory):
    processor = StubProcessor()
    client = app_factory(processor).test_client()
    login(client)
    response = post_image(client, "/admin/vehicles/from-image")
    assert response.status_code == 200
    assert b"Detected plate crop" in response.data
    assert b'name="plate_number"' in response.data
    assert b'value="3003"' in response.data
    assert processor.existed_during_processing
    assert not processor.path.exists()
    assert not client.application.extensions["anpr_store"].list_public("3003")


def test_correcting_ocr_before_confirmation_creates_corrected_record(client):
    login(client)
    response = post(
        client,
        "/admin/vehicles/from-image/confirm",
        vehicle_form("CORRECT7"),
    )
    assert response.status_code == 302
    assert client.application.extensions["anpr_store"].list_public("CORRECT7")


@pytest.mark.parametrize(
    ("review", "message"),
    [
        (
            ImageReview(
                "data:image/jpeg;base64,original",
                None,
                "",
                None,
                "No plate-like region was found. Enter the plate manually.",
            ),
            b"No plate-like region was found",
        ),
        (
            ImageReview(
                "data:image/jpeg;base64,original",
                "data:image/jpeg;base64,crop",
                "",
                "Contour detection",
                "OCR returned no usable number. Enter the plate manually.",
            ),
            b"OCR returned no usable number",
        ),
    ],
)
def test_failed_image_analysis_allows_manual_entry_without_saving(
    app_factory, review, message
):
    client = app_factory(StubProcessor(review)).test_client()
    login(client)
    response = post_image(client, "/admin/vehicles/from-image")
    assert message in response.data
    assert b'name="plate_number"' in response.data
    assert not client.application.extensions["anpr_store"].list_public("3003")


def test_duplicate_image_confirmation_is_rejected(client):
    login(client)
    response = post(
        client,
        "/admin/vehicles/from-image/confirm",
        vehicle_form("1001"),
    )
    assert response.status_code == 400
    assert b"already registered" in response.data


@pytest.mark.parametrize(
    ("ocr_text", "expected"),
    [("1001", b"ALLOWED"), ("1002", b"NOT ALLOWED")],
)
def test_demo_uses_exact_sqlite_match(app_factory, runtime_config, ocr_text, expected):
    client = app_factory(pipeline_processor(runtime_config, ocr_text)).test_client()
    response = post_image(client)
    assert expected in response.data


def test_inactive_new_and_deleted_vehicle_authorization(app_factory, runtime_config):
    client = app_factory(pipeline_processor(runtime_config, "3003")).test_client()
    store = client.application.extensions["anpr_store"]
    vehicle_id = store.create("3003", "Jordan Fiction", "visitor", True)
    assert b"ALLOWED" in post_image(client).data
    store.set_active(vehicle_id, False)
    assert b"NOT ALLOWED" in post_image(client).data
    store.set_active(vehicle_id, True)
    assert b"ALLOWED" in post_image(client).data
    store.apply_checkpoint("3003", "exit")
    store.delete(vehicle_id)
    assert b"NOT ALLOWED" in post_image(client).data


def test_upload_validation_and_health(client):
    invalid = post_image(client, filename="vehicle.gif")
    assert invalid.status_code == 400
    assert b"Please upload a JPG, JPEG, or PNG file." in invalid.data
    corrupted = post_image(client, payload=b"not really an image")
    assert corrupted.status_code == 400
    assert b"corrupted or is not an image" in corrupted.data
    assert client.get("/health").json["status"] == "ok"


def test_oversized_upload_is_rejected(client):
    response = post_image(client, payload=b"x" * (8 * 1024 * 1024 + 1))
    assert response.status_code == 413
    assert b"larger than the 8 MB upload limit" in response.data


def test_demo_distinguishes_no_region_from_empty_ocr(app_factory, runtime_config):
    no_region = WebProcessor(
        runtime_config,
        detector=lambda _frame: (None, None, None),
        ocr_loader=lambda: pytest.fail("OCR must not load without a plate"),
    )
    response = post_image(app_factory(no_region).test_client())
    assert b"No plate-like region was found" in response.data

    empty_ocr = pipeline_processor(runtime_config, "")
    response = post_image(app_factory(empty_ocr).test_client())
    assert b"OCR returned no usable number" in response.data


def pipeline_processor(runtime_config, ocr_text):
    crop = np.zeros((20, 40, 3), dtype=np.uint8)

    def detector(_frame):
        return crop, (10, 10, 40, 20), "Contour detection"

    def ocr(_crop, _reader, confidence_threshold):
        assert confidence_threshold == runtime_config.ocr_confidence_threshold
        return ocr_text

    return WebProcessor(
        runtime_config,
        detector=detector,
        ocr=ocr,
        ocr_loader=lambda: object(),
    )


def test_store_fixture_uses_sqlite(client):
    assert isinstance(client.application.extensions["anpr_store"], VehicleStore)


def allowed_stub_result(plate="1001"):
    return ProcessingResult(
        "ALLOWED",
        "Exact test match",
        plate,
        {
            "matched_plate": plate,
            "name": "Alex Example",
            "type": "student",
            "distance": 0,
        },
        "data:image/jpeg;base64,",
    )


def test_demo_entry_and_exit_results_are_clear_and_idempotent(app_factory):
    processor = StubProcessor()
    processor.process = lambda _path, _database: allowed_stub_result()
    entry_client = app_factory(processor, ANPR_CHECKPOINT_MODE="entry").test_client()
    assert b"Entry recorded" in post_image(entry_client).data
    assert b"Already inside" in post_image(entry_client).data
    assert entry_client.application.extensions["anpr_store"].count_visits() == 1

    exit_client = app_factory(processor, ANPR_CHECKPOINT_MODE="exit").test_client()
    assert b"No active entry found" in post_image(exit_client).data
    store = exit_client.application.extensions["anpr_store"]
    store.apply_checkpoint("1001", "entry")
    assert b"Exit recorded" in post_image(exit_client).data


def test_selectable_checkpoint_requires_valid_action(app_factory):
    client = app_factory(
        StubProcessor(), ANPR_CHECKPOINT_MODE="selectable"
    ).test_client()
    missing = post_image(client)
    assert missing.status_code == 400
    assert b"Select a valid Entry or Exit" in missing.data
    invalid = post(
        client,
        "/analyse",
        {
            "checkpoint_action": "sideways",
            "vehicle_image": (BytesIO(image_bytes()), "vehicle.png"),
        },
        content_type="multipart/form-data",
    )
    assert invalid.status_code == 400
    assert b"Select a valid Entry or Exit" in invalid.data


def test_access_history_filters_timezone_duration_and_metrics(app_factory):
    app = app_factory(StubProcessor(), ANPR_TIMEZONE="Asia/Tokyo")
    client = app.test_client()
    store = app.extensions["anpr_store"]
    store.apply_checkpoint("1001", "entry", now="2026-07-30T15:00:00+00:00")
    store.apply_checkpoint("1001", "exit", now="2026-07-30T16:30:00+00:00")
    store.apply_checkpoint("2002", "entry", now="2026-07-31T01:00:00+00:00")

    response = client.get("/access-history")
    assert response.status_code == 200
    assert b"2026-07-31 00:00" in response.data
    assert b"1h 30m" in response.data
    assert b"Inside" in response.data and b"Exited" in response.data
    assert b"Alex Example" in client.get("/access-history?q=1001").data
    assert b"Casey Sample" not in client.get("/access-history?status=exited").data
    dated = client.get("/access-history?date_from=2026-07-31&date_to=2026-07-31")
    assert b"Alex Example" in dated.data and b"Casey Sample" in dated.data

    login(client)
    admin_view = client.get("/access-history")
    assert b"Currently inside" in admin_view.data
    assert b"Average visit" in admin_view.data


def test_access_history_paginates(app_factory):
    app = app_factory(StubProcessor())
    store = app.extensions["anpr_store"]
    for day in range(1, 13):
        store.apply_checkpoint("1001", "entry", now=f"2026-07-{day:02d}T00:00:00+00:00")
        store.apply_checkpoint("1001", "exit", now=f"2026-07-{day:02d}T01:00:00+00:00")
    response = app.test_client().get("/access-history?page=2")
    assert response.status_code == 200
    assert b'aria-current="page"' in response.data


def test_camera_endpoint_requires_login_and_stable_confirmation(app_factory):
    processor = StubProcessor()
    app = app_factory(
        processor,
        CAMERA_SAMPLE_INTERVAL_SECONDS=0,
        CAMERA_STABLE_SAMPLES=2,
    )
    client = app.test_client()
    assert post_camera_frame(client).status_code == 302
    login(client)

    first = post_camera_frame(client)
    assert first.status_code == 200
    assert first.json["status"] == "sampling"
    assert not app.extensions["anpr_store"].list_public("3003")
    second = post_camera_frame(client)
    assert second.json["status"] == "review"
    assert second.json["plate"] == "3003"
    assert processor.existed_during_processing
    assert not processor.path.exists()
    assert not app.extensions["anpr_store"].list_public("3003")

    saved = post(
        client,
        "/admin/vehicles/from-camera/confirm",
        vehicle_form("CORRECT9"),
    )
    assert saved.status_code == 302
    assert app.extensions["anpr_store"].list_public("CORRECT9")


def test_camera_rejects_corrupted_duplicate_and_oversized_frames(app_factory):
    app = app_factory(
        StubProcessor(),
        CAMERA_SAMPLE_INTERVAL_SECONDS=0,
        CAMERA_FRAME_MAX_BYTES=1024,
    )
    client = app.test_client()
    login(client)
    corrupted = post_camera_frame(client, payload=b"bad")
    assert corrupted.status_code == 400
    oversized = post_camera_frame(client, payload=b"x" * 2048)
    assert oversized.status_code == 413
    duplicate = post(
        client,
        "/admin/vehicles/from-camera/confirm",
        vehicle_form("1001"),
    )
    assert duplicate.status_code == 400
    assert b"already registered" in duplicate.data


def test_theme_and_camera_fallbacks_render(client):
    base = client.get("/")
    assert b'class="theme-toggle"' in base.data
    assert b"prefers-color-scheme" in base.data
    assert b"localStorage" in base.data
    login(client)
    camera = client.get("/admin/vehicles/from-camera")
    assert camera.status_code == 200
    assert b"getUserMedia" not in camera.data
    javascript = client.get("/static/camera.js")
    assert b"getUserMedia" in javascript.data
    assert b"image upload" in camera.data and b"manual registration" in camera.data


def test_theme_aware_button_colors_are_defined_and_used(client):
    stylesheet = client.get("/static/styles.css")
    assert stylesheet.status_code == 200
    css = stylesheet.get_data(as_text=True)
    for variable in (
        "--control-bg:",
        "--control-text:",
        "--control-hover-bg:",
        "--control-active-bg:",
        "--secondary-control-bg:",
        "--secondary-control-text:",
        "--disabled-control-bg:",
        "--disabled-control-text:",
        "--danger-control-bg:",
        "--danger-control-hover-bg:",
        "--focus-ring:",
    ):
        assert css.count(variable) == 2
    assert "background: var(--control-bg); color: var(--control-text);" in css
    assert "background: var(--ink); color: white;" not in css
    assert 'button:disabled, button[aria-disabled="true"]' in css
    assert ".secondary:hover" in css
    assert ".danger:hover" in css
    assert ".actions a:hover" in css
    assert ".delivery-badge.sent" in css
    assert ".occupancy-card" in css


def test_plate_badges_have_explicit_theme_colors_and_template_coverage(client):
    css = client.get("/static/styles.css").get_data(as_text=True)
    for variable in (
        "--plate-badge-bg:",
        "--plate-badge-text:",
        "--plate-badge-border:",
    ):
        assert css.count(variable) == 2
    dark_theme = css.split(':root[data-theme="dark"]', 1)[1].split("}", 1)[0]
    assert "--plate-badge-text: #122025;" in dark_theme
    assert ".plate-badge {" in css
    assert "color: var(--plate-badge-text);" in css

    templates = Path("anpr_web/templates")
    for relative_path in (
        "vehicles.html",
        "history.html",
        "result.html",
        "admin/dashboard.html",
        "admin/vehicles.html",
        "admin/delete_confirm.html",
        "admin/audit_log.html",
    ):
        markup = (templates / relative_path).read_text(encoding="utf-8")
        assert "plate-badge" in markup


def test_plate_badges_render_on_public_and_admin_pages(client):
    assert b'class="plate-badge"' in client.get("/registered-vehicles").data
    login(client)
    assert b'class="plate-badge"' in client.get("/admin").data
    assert b'class="plate-badge"' in client.get("/admin/vehicles").data
    vehicle_id = client.application.extensions["anpr_store"].list_all()[0]["id"]
    delete_page = client.get(f"/admin/vehicles/{vehicle_id}/delete")
    assert delete_page.data.count(b'class="plate-badge"') == 2


@pytest.mark.parametrize(
    "path",
    (
        "/",
        "/admin/login",
        "/registered-vehicles",
        "/access-history",
        "/privacy",
    ),
)
def test_main_public_templates_render(client, path):
    assert client.get(path).status_code == 200


@pytest.mark.parametrize(
    "path",
    (
        "/admin",
        "/admin/vehicles",
        "/admin/vehicles/add",
        "/admin/vehicles/from-image",
        "/admin/vehicles/from-camera",
        "/admin/audit-log",
    ),
)
def test_main_admin_templates_render(client, path):
    login(client)
    assert client.get(path).status_code == 200


class RecordingMailBackend:
    def __init__(self, store=None, fail=False):
        self.store = store
        self.fail = fail
        self.messages = []
        self.exit_was_committed = False

    def send(self, message):
        if self.store:
            records, _ = self.store.history(status="exited")
            self.exit_was_committed = bool(records and records[0]["exited_at"])
        if self.fail:
            raise RuntimeError("provider details must stay private")
        self.messages.append(message)


def email_app(app_factory, backend, **overrides):
    config = {
        "ANPR_CHECKPOINT_MODE": "exit",
        "ANPR_EMAIL_ENABLED": True,
        "ANPR_EMAIL_BACKEND": "smtp",
        "ANPR_SMTP_HOST": "smtp.example.test",
        "ANPR_EMAIL_FROM": "parking@example.test",
        "ANPR_APPLICATION_NAME": "Fictional Example Campus",
        "ANPR_MAIL_BACKEND": backend,
    }
    config.update(overrides)
    processor = StubProcessor()
    processor.process = lambda _path, _database: allowed_stub_result()
    app = app_factory(processor, **config)
    backend.store = app.extensions["anpr_store"]
    return app


def enable_seed_vehicle_email(store, enabled=True):
    vehicle = next(row for row in store.list_all() if row["plate_number"] == "1001")
    store.update_vehicle(
        vehicle["id"],
        plate_number=vehicle["plate_number"],
        display_name=vehicle["display_name"],
        category=vehicle["category"],
        is_active=True,
        email="owner@example.test",
        email_notifications_enabled=enabled,
    )
    return vehicle["id"]


def test_vehicle_email_validation_privacy_and_shared_registration(client):
    login(client)
    for path in (
        "/admin/vehicles/add",
        "/admin/vehicles/from-image/confirm",
        "/admin/vehicles/from-camera/confirm",
    ):
        data = vehicle_form(f"BAD{len(path)}")
        data["email"] = "not-an-email"
        data["email_notifications_enabled"] = "1"
        response = post(client, path, data)
        assert response.status_code in {200, 400}
        assert b"Enter a valid email address" in response.data

    blank = vehicle_form("BLANK1")
    blank["email"] = ""
    assert post(client, "/admin/vehicles/add", blank).status_code == 302
    rejected = vehicle_form("BLANK2")
    rejected["email_notifications_enabled"] = "1"
    response = post(client, "/admin/vehicles/add", rejected)
    assert b"email address is required" in response.data

    accepted = vehicle_form("MAIL2")
    accepted.update(email="Person@EXAMPLE.COM", email_notifications_enabled="1")
    assert post(client, "/admin/vehicles/add", accepted).status_code == 302
    stored = client.application.extensions["anpr_store"].list_public("MAIL2")[0]
    assert "email" not in stored.keys()
    public = client.get("/registered-vehicles")
    assert b"Person@example.com" not in public.data


def test_successful_exit_sends_once_after_commit_and_masks_delivery(app_factory):
    backend = RecordingMailBackend()
    app = email_app(app_factory, backend)
    client = app.test_client()
    store = app.extensions["anpr_store"]
    vehicle_id = enable_seed_vehicle_email(store)
    entered = "2026-07-30T15:00:00+00:00"
    store.apply_checkpoint("1001", "entry", now=entered)

    response = post_image(client)
    assert b"Visit summary email sent" in response.data
    assert backend.exit_was_committed
    assert len(backend.messages) == 1
    delivery = store.delivery_for_visit(
        store.history(status="exited")[0][0]["visit_id"]
    )
    assert delivery["vehicle_id"] == vehicle_id
    assert delivery["status"] == "sent"
    assert delivery["recipient_email_masked"] == "o****@example.test"
    assert "owner@example.test" not in delivery["recipient_email_masked"]
    message = backend.messages[0]
    content = str(message)
    assert "Fictional Example Campus" in content
    assert "Visit completed" not in content
    assert "1001" not in content
    assert "credentials" not in content
    assert "session" not in content
    assert "image" not in content
    assert b"No active entry found" in post_image(client).data
    assert len(backend.messages) == 1
    assert store.count_deliveries() == 1


@pytest.mark.parametrize(
    ("preference", "email_enabled", "expected_status"),
    [(False, True, "skipped"), (True, False, "skipped")],
)
def test_disabled_email_modes_never_contact_backend(
    app_factory, preference, email_enabled, expected_status
):
    backend = RecordingMailBackend()
    app = email_app(app_factory, backend, ANPR_EMAIL_ENABLED=email_enabled)
    client = app.test_client()
    store = app.extensions["anpr_store"]
    enable_seed_vehicle_email(store, preference)
    checkpoint = store.apply_checkpoint_detail("1001", "entry")
    response = post_image(client)
    assert b"Email notification disabled" in response.data
    assert backend.messages == []
    delivery = store.delivery_for_visit(checkpoint.visit_id)
    assert delivery["status"] == expected_status


def test_email_failure_preserves_exit_and_admin_retry_is_idempotent(app_factory):
    backend = RecordingMailBackend(fail=True)
    app = email_app(app_factory, backend)
    client = app.test_client()
    store = app.extensions["anpr_store"]
    enable_seed_vehicle_email(store)
    checkpoint = store.apply_checkpoint_detail(
        "1001", "entry", now="2026-07-30T15:00:00+00:00"
    )
    response = post_image(client)
    assert b"Visit recorded, but email delivery failed" in response.data
    delivery = store.delivery_for_visit(checkpoint.visit_id)
    assert delivery["status"] == "failed"
    visit = store.visit_for_notification(checkpoint.visit_id)
    assert visit["exited_at"] is not None
    assert store.apply_checkpoint("1001", "entry") == "entry_recorded"

    retry_path = f"/admin/notifications/{delivery['id']}/retry"
    assert client.post(retry_path).status_code == 400
    backend.fail = False
    login(client)
    assert post(client, retry_path).status_code == 302
    assert store.get_delivery(delivery["id"])["status"] == "sent"
    assert len(backend.messages) == 1
    assert post(client, retry_path).status_code == 409
    assert len(backend.messages) == 1


def test_http_api_failure_preserves_recorded_exit(app_factory):
    def failed_https_request(_request, timeout):
        assert timeout == 5
        raise urllib.error.URLError("provider unavailable")

    config = {
        "ANPR_EMAIL_API_URL": "https://mail.example.test/send",
        "ANPR_EMAIL_API_KEY": "test-key",
        "ANPR_EMAIL_TIMEOUT_SECONDS": 5,
    }
    backend = HTTPAPIBackend(config, opener=failed_https_request)
    app = email_app(
        app_factory,
        backend,
        ANPR_EMAIL_BACKEND="http-api",
        ANPR_EMAIL_API_URL=config["ANPR_EMAIL_API_URL"],
        ANPR_EMAIL_API_KEY=config["ANPR_EMAIL_API_KEY"],
    )
    client = app.test_client()
    store = app.extensions["anpr_store"]
    enable_seed_vehicle_email(store)
    checkpoint = store.apply_checkpoint_detail("1001", "entry")

    response = post_image(client)

    assert b"Visit recorded, but email delivery failed" in response.data
    assert store.visit_for_notification(checkpoint.visit_id)["exited_at"] is not None
    assert store.delivery_for_visit(checkpoint.visit_id)["status"] == "failed"


def test_occupancy_capacity_and_deleted_vehicle_rules(app_factory):
    app = app_factory(StubProcessor(), ANPR_PARKING_CAPACITY=4)
    store = app.extensions["anpr_store"]
    first = store.apply_checkpoint_detail("1001", "entry")
    store.apply_checkpoint("2002", "entry")
    assert store.occupancy_count() == 2
    store.apply_checkpoint("2002", "exit")
    assert store.occupancy_count() == 1
    vehicle = store.get(first.vehicle_id)
    with pytest.raises(ValueError, match="1001"):
        store.delete(vehicle["id"])
    store.apply_checkpoint("1001", "exit")
    store.delete(vehicle["id"])
    assert store.occupancy_count() == 0
    client = app.test_client()
    login(client)
    dashboard = client.get("/admin")
    assert b"4" in dashboard.data
    assert b"spaces available" in dashboard.data
    with pytest.raises(ValueError, match="positive integer"):
        app_factory(StubProcessor(), ANPR_PARKING_CAPACITY=0)


def test_admin_csv_export_filters_timezone_privacy_and_formula_safety(app_factory):
    app = app_factory(StubProcessor(), ANPR_TIMEZONE="Asia/Tokyo")
    store = app.extensions["anpr_store"]
    vehicle_id = store.create("FORM1", "=Formula Example", "visitor")
    store.apply_checkpoint("FORM1", "entry", now="2026-07-30T15:00:00+00:00")
    store.apply_checkpoint("FORM1", "exit", now="2026-07-30T16:00:00+00:00")
    store.delete(vehicle_id)
    client = app.test_client()
    assert client.get("/admin/access-history/export.csv").status_code == 302
    login(client)
    response = client.get("/admin/access-history/export.csv?q=FORM1&status=exited")
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("text/csv")
    assert b"'=Formula Example" in response.data
    assert b"Deleted vehicle record" in response.data
    assert b"2026-07-31 00:00" in response.data
    assert b"email" in response.data.lower()
    assert b"example.test" not in response.data
    assert b"vehicle_id" not in response.data


def test_audit_log_records_required_actions_without_sensitive_values(client):
    login(client)
    data = vehicle_form("AUDIT1")
    data.update(email="audit@example.test", email_notifications_enabled="1")
    assert post(client, "/admin/vehicles/add", data).status_code == 302
    store = client.application.extensions["anpr_store"]
    vehicle = next(row for row in store.list_all() if row["plate_number"] == "AUDIT1")
    post(client, f"/admin/vehicles/{vehicle['id']}/deactivate")
    edit = vehicle_form("AUDIT1")
    edit["email"] = "audit@example.test"
    post(client, f"/admin/vehicles/{vehicle['id']}/edit", edit)
    page = client.get("/admin/audit-log")
    assert page.status_code == 200
    assert b"Vehicle Registered Manual" in page.data
    assert b"Vehicle Deactivated" in page.data
    assert b"Notification Preference Changed" in page.data
    assert b"audit@example.test" not in page.data
    anonymous = client.application.test_client()
    assert anonymous.get("/admin/audit-log").status_code == 302


def test_invalid_email_configuration_warns_without_blocking_startup(app_factory):
    app = app_factory(
        StubProcessor(),
        ANPR_EMAIL_ENABLED=True,
        ANPR_SMTP_HOST="",
        ANPR_EMAIL_FROM="",
        ANPR_SMTP_PORT="invalid",
    )
    assert not app.config["ANPR_EMAIL_READY"]
    assert "positive integer" in app.config["ANPR_EMAIL_CONFIGURATION_WARNING"]
    client = app.test_client()
    login(client)
    assert b"ANPR_SMTP_PORT" in client.get("/admin").data
