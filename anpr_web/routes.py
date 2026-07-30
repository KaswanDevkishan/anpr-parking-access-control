"""Server-rendered public and admin routes."""

import csv
import io
import math
import time
from datetime import datetime, timedelta, timezone
from datetime import time as datetime_time

from flask import (
    Blueprint,
    Response,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from matcher import normalise_plate

from .database import CATEGORIES, VehicleCurrentlyInside
from .mail import VisitSummary, mask_email, mask_plate, parse_utc
from .processing import InvalidImage, remove_temporary_file, save_validated_upload
from .security import (
    LOGIN_SESSION_KEY,
    admin_required,
    authenticate_admin,
    is_safe_redirect,
    login_locked,
    record_login_failure,
)
from .vehicles import save_vehicle, vehicle_form_values

web = Blueprint("web", __name__)
HISTORY_PAGE_SIZE = 10
AUDIT_PAGE_SIZE = 20


def _store():
    return current_app.extensions["anpr_store"]


@web.get("/")
def index():
    return render_template("index.html")


@web.post("/analyse")
def analyse():
    action, action_error = _checkpoint_action(request.form.get("checkpoint_action"))
    if action_error:
        return render_template("index.html", error=action_error), 400
    upload = request.files.get("vehicle_image")
    if upload is None or not upload.filename:
        return render_template("index.html", error="Choose an image to analyse."), 400

    temporary_path = None
    try:
        temporary_path = save_validated_upload(
            upload, current_app.config["UPLOAD_TEMP_DIR"]
        )
        result = current_app.extensions["anpr_processor"].process(
            temporary_path,
            _store().active_match_database(),
        )
    except InvalidImage as exc:
        return render_template("index.html", error=str(exc)), 400
    finally:
        remove_temporary_file(temporary_path)

    checkpoint_result = None
    notification_result = None
    if result.status == "ALLOWED" and result.match:
        checkpoint = _store().apply_checkpoint_detail(
            result.match["matched_plate"], action, "web-demo"
        )
        checkpoint_result = checkpoint.status
        if checkpoint.status == "exit_recorded":
            notification_result = _deliver_exit_summary(checkpoint.visit_id)
    return render_template(
        "result.html",
        result=result,
        checkpoint_result=checkpoint_result,
        notification_result=notification_result,
    )


@web.get("/registered-vehicles")
def registered_vehicles():
    search = request.args.get("q", "").strip()
    category = request.args.get("category", "").strip().lower()
    records = _store().list_public(search, category)
    return render_template(
        "vehicles.html",
        records=records,
        search=search,
        selected_category=category,
        categories=CATEGORIES,
    )


@web.get("/demo-vehicles")
def demo_vehicles_redirect():
    return redirect(url_for("web.registered_vehicles"))


@web.get("/architecture")
def architecture():
    return render_template("architecture.html")


@web.get("/privacy")
def privacy():
    return render_template("privacy.html")


@web.get("/access-history")
def access_history():
    search = request.args.get("q", "").strip()
    status = request.args.get("status", "").strip().lower()
    if status not in {"", "inside", "exited"}:
        status = ""
    date_from = _valid_date(request.args.get("date_from", ""))
    date_to = _valid_date(request.args.get("date_to", ""))
    try:
        page = max(1, int(request.args.get("page", "1")))
    except ValueError:
        page = 1
    date_from_utc, date_to_utc = _date_bounds_utc(date_from, date_to)
    records, total = _store().history(
        search, status, date_from_utc, date_to_utc, page, HISTORY_PAGE_SIZE
    )
    pages = max(1, math.ceil(total / HISTORY_PAGE_SIZE))
    if page > pages:
        page = pages
        records, total = _store().history(
            search, status, date_from_utc, date_to_utc, page, HISTORY_PAGE_SIZE
        )
    view_records = [_visit_view(record) for record in records]
    is_admin = bool(session.get(LOGIN_SESSION_KEY))
    metrics = _history_metrics() if is_admin else None
    return render_template(
        "history.html",
        records=view_records,
        search=search,
        selected_status=status,
        date_from=date_from,
        date_to=date_to,
        page=page,
        pages=pages,
        total=total,
        metrics=metrics,
        is_admin=is_admin,
    )


@web.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if session.get(LOGIN_SESSION_KEY):
        return redirect(url_for("web.admin_dashboard"))
    error = None
    if request.method == "POST":
        client_key = request.remote_addr or "unknown"
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if login_locked(client_key):
            error = "Login failed. Please check your credentials or try again later."
            return render_template("admin/login.html", error=error), 429
        if authenticate_admin(username, password):
            session.clear()
            session[LOGIN_SESSION_KEY] = True
            attempts = current_app.extensions["anpr_login_attempts"]
            attempts.pop(client_key, None)
            target = request.form.get("next", "")
            if is_safe_redirect(target):
                return redirect(target)
            return redirect(url_for("web.admin_dashboard"))
        record_login_failure(client_key)
        error = "Login failed. Please check your credentials or try again later."
    return render_template("admin/login.html", error=error)


@web.post("/admin/logout")
@admin_required
def admin_logout():
    session.clear()
    return redirect(url_for("web.index"))


@web.get("/admin")
@admin_required
def admin_dashboard():
    occupancy = _occupancy_summary()
    return render_template(
        "admin/dashboard.html",
        counts=_store().counts(),
        recent=_store().list_recent(),
        occupancy=occupancy,
        email_warning=current_app.config["ANPR_EMAIL_CONFIGURATION_WARNING"],
    )


@web.get("/admin/vehicles")
@admin_required
def admin_vehicles():
    return render_template("admin/vehicles.html", records=_store().list_all())


@web.route("/admin/vehicles/add", methods=["GET", "POST"])
@admin_required
def admin_add_vehicle():
    values = vehicle_form_values(request.form)
    errors = []
    if request.method == "POST":
        _vehicle_id, errors = save_vehicle(_store(), values)
        if not errors:
            _store().audit(
                "vehicle_registered_manual",
                "vehicle",
                values["plate_number"],
                "Manual vehicle registration completed.",
            )
            flash("Vehicle registered.", "success")
            return redirect(url_for("web.admin_vehicles"))
    return render_template(
        "admin/vehicle_form.html",
        values=values,
        errors=errors,
        categories=CATEGORIES,
    )


@web.post("/admin/vehicles/<int:vehicle_id>/deactivate")
@admin_required
def admin_deactivate_vehicle(vehicle_id):
    vehicle = _store().get(vehicle_id)
    if vehicle and _store().set_active(vehicle_id, False):
        _store().audit(
            "vehicle_deactivated",
            "vehicle",
            vehicle["plate_number"],
            "Vehicle access deactivated.",
        )
    flash("Vehicle deactivated.", "success")
    return redirect(url_for("web.admin_vehicles"))


@web.post("/admin/vehicles/<int:vehicle_id>/reactivate")
@admin_required
def admin_reactivate_vehicle(vehicle_id):
    vehicle = _store().get(vehicle_id)
    if vehicle and _store().set_active(vehicle_id, True):
        _store().audit(
            "vehicle_activated",
            "vehicle",
            vehicle["plate_number"],
            "Vehicle access activated.",
        )
    flash("Vehicle reactivated.", "success")
    return redirect(url_for("web.admin_vehicles"))


@web.route("/admin/vehicles/<int:vehicle_id>/delete", methods=["GET", "POST"])
@admin_required
def admin_delete_vehicle(vehicle_id):
    vehicle = _store().get(vehicle_id)
    if vehicle is None:
        return ("Vehicle not found.", 404)
    if request.method == "POST":
        if request.form.get("confirmation") != vehicle["plate_number"]:
            return render_template(
                "admin/delete_confirm.html",
                vehicle=vehicle,
                error="Type the plate number exactly to confirm permanent deletion.",
            ), 400
        try:
            deleted = _store().delete(vehicle_id)
        except VehicleCurrentlyInside:
            return render_template(
                "admin/delete_confirm.html",
                vehicle=vehicle,
                error=(
                    "This vehicle is currently inside. "
                    "Record its exit before deleting it."
                ),
            ), 409
        if not deleted:
            return ("Vehicle not found.", 404)
        flash(
            "Vehicle permanently deleted; retained visit history is unchanged.",
            "success",
        )
        return redirect(url_for("web.admin_vehicles"))
    return render_template("admin/delete_confirm.html", vehicle=vehicle)


@web.route("/admin/vehicles/from-image", methods=["GET", "POST"])
@admin_required
def admin_vehicle_from_image():
    if request.method == "GET":
        return render_template("admin/image_upload.html")
    upload = request.files.get("vehicle_image")
    if upload is None or not upload.filename:
        return render_template(
            "admin/image_upload.html", error="Choose an image to analyse."
        ), 400
    temporary_path = None
    try:
        temporary_path = save_validated_upload(
            upload, current_app.config["UPLOAD_TEMP_DIR"]
        )
        review = current_app.extensions["anpr_processor"].review(temporary_path)
    except InvalidImage as exc:
        return render_template("admin/image_upload.html", error=str(exc)), 400
    finally:
        remove_temporary_file(temporary_path)
    values = vehicle_form_values({"plate_number": review.ocr_text, "is_active": "1"})
    return render_template(
        "admin/image_review.html",
        review=review,
        values=values,
        errors=[],
        categories=CATEGORIES,
    )


@web.post("/admin/vehicles/from-image/confirm")
@admin_required
def admin_confirm_image_vehicle():
    values = vehicle_form_values(request.form)
    _vehicle_id, errors = save_vehicle(_store(), values)
    if not errors:
        _store().audit(
            "vehicle_registered_image",
            "vehicle",
            values["plate_number"],
            "Image-assisted vehicle registration confirmed.",
        )
        flash("Image-assisted vehicle registration confirmed.", "success")
        return redirect(url_for("web.admin_vehicles"))
    return render_template(
        "admin/image_review.html",
        review=None,
        values=values,
        errors=errors,
        categories=CATEGORIES,
    ), 400


@web.get("/admin/vehicles/from-camera")
@admin_required
def admin_vehicle_from_camera():
    return render_template("admin/camera.html")


@web.post("/admin/vehicles/from-camera/frame")
@admin_required
def admin_camera_frame():
    now = time.monotonic()
    state = current_app.extensions["anpr_camera_samples"].setdefault(
        session.sid if hasattr(session, "sid") else session.get("_csrf_token"),
        {"started": now, "last": 0.0, "plate": "", "count": 0},
    )
    if now - state["started"] > current_app.config["CAMERA_SESSION_TIMEOUT_SECONDS"]:
        state.update(started=now, last=0.0, plate="", count=0)
        return jsonify(status="timeout", message="Detection timed out. Try again."), 408
    if now - state["last"] < current_app.config["CAMERA_SAMPLE_INTERVAL_SECONDS"]:
        return jsonify(status="wait", message="Sampling is limited for privacy."), 429
    state["last"] = now
    upload = request.files.get("frame")
    if upload is None:
        return jsonify(status="error", message="A camera frame is required."), 400
    payload_length = request.content_length or 0
    if payload_length > current_app.config["CAMERA_FRAME_MAX_BYTES"]:
        return jsonify(status="error", message="Camera frame is too large."), 413
    temporary_path = None
    try:
        temporary_path = save_validated_upload(
            upload, current_app.config["UPLOAD_TEMP_DIR"]
        )
        review = current_app.extensions["anpr_processor"].review(temporary_path)
    except InvalidImage as exc:
        return jsonify(status="error", message=str(exc)), 400
    finally:
        remove_temporary_file(temporary_path)

    plate = normalise_plate(review.ocr_text)
    manual = request.form.get("manual") == "1"
    if plate and plate == state["plate"]:
        state["count"] += 1
    elif plate:
        state.update(plate=plate, count=1)
    else:
        state.update(plate="", count=0)
    stable = state["count"] >= current_app.config["CAMERA_STABLE_SAMPLES"]
    if not (manual or stable):
        return jsonify(
            status="sampling",
            plate=plate,
            stable_count=state["count"],
            message=review.error or "Waiting for a stable plate reading.",
        )
    state.update(started=now, last=0.0, plate="", count=0)
    return jsonify(
        status="review",
        plate=plate,
        original_image=review.original_image,
        cropped_image=review.cropped_image,
        detection_method=review.detection_method,
        message=review.error or "Review and explicitly confirm before saving.",
    )


@web.post("/admin/vehicles/from-camera/confirm")
@admin_required
def admin_confirm_camera_vehicle():
    values = vehicle_form_values(request.form)
    _vehicle_id, errors = save_vehicle(_store(), values)
    if not errors:
        _store().audit(
            "vehicle_registered_camera",
            "vehicle",
            values["plate_number"],
            "Browser-camera vehicle registration confirmed.",
        )
        flash("Camera-assisted vehicle registration confirmed.", "success")
        return redirect(url_for("web.admin_vehicles"))
    return render_template(
        "admin/camera.html",
        values=values,
        errors=errors,
        categories=CATEGORIES,
        show_review=True,
    ), 400


@web.route("/admin/vehicles/<int:vehicle_id>/edit", methods=["GET", "POST"])
@admin_required
def admin_edit_vehicle(vehicle_id):
    vehicle = _store().get(vehicle_id)
    if vehicle is None:
        return ("Vehicle not found.", 404)
    values = vehicle_form_values(request.form, vehicle)
    errors = []
    if request.method == "POST":
        preference_changed = (
            bool(vehicle["email_notifications_enabled"])
            != values["email_notifications_enabled"]
        )
        _saved_id, errors = save_vehicle(_store(), values, vehicle_id)
        if not errors:
            if preference_changed:
                _store().audit(
                    "notification_preference_changed",
                    "vehicle",
                    values["plate_number"],
                    "Visit-summary notification preference changed.",
                )
            flash("Vehicle details updated.", "success")
            return redirect(url_for("web.admin_vehicles"))
    return render_template(
        "admin/vehicle_form.html",
        values=values,
        errors=errors,
        categories=CATEGORIES,
        editing=True,
    )


@web.get("/admin/access-history/export.csv")
@admin_required
def admin_history_export():
    search, status, date_from, date_to, date_from_utc, date_to_utc = (
        _history_filter_values()
    )
    records = _store().history_export(search, status, date_from_utc, date_to_utc)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        (
            "Plate number",
            "Display name",
            "Category",
            "Entry time",
            "Exit time",
            "Duration",
            "Visit status",
            "Vehicle record status",
            "Email delivery status",
        )
    )
    for record in records:
        view = _visit_view(record)
        writer.writerow(
            _csv_safe(value)
            for value in (
                view["plate_number"],
                view["display_name"],
                view["category"],
                view["entry_time"],
                view["exit_time"],
                view["duration"],
                view["status"],
                view["vehicle_record_status"] or "Current vehicle record",
                view["delivery_status"],
            )
        )
    filename = datetime.now(timezone.utc).strftime("access-history-%Y%m%d-%H%M%S.csv")
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@web.post("/admin/notifications/<int:delivery_id>/retry")
@admin_required
def admin_retry_notification(delivery_id):
    delivery = _store().delivery_context(delivery_id)
    if delivery is None:
        return ("Notification delivery not found.", 404)
    if delivery["status"] == "sent":
        return ("A successful notification cannot be resent.", 409)
    if delivery["status"] != "failed":
        return ("Only failed notifications can be retried.", 409)
    result = _attempt_delivery(delivery)
    action = "email_retry_succeeded" if result == "sent" else "email_retry_failed"
    _store().audit(
        action,
        "visit",
        mask_plate(delivery["plate_number"]),
        "Exit-summary email retry completed without storing recipient details.",
    )
    flash(
        "Visit summary email sent."
        if result == "sent"
        else "Email retry failed; the recorded exit is unchanged.",
        "success" if result == "sent" else "error",
    )
    return redirect(url_for("web.access_history"))


@web.get("/admin/audit-log")
@admin_required
def admin_audit_log():
    try:
        page = max(1, int(request.args.get("page", "1")))
    except ValueError:
        page = 1
    records, total = _store().audit_log(page, AUDIT_PAGE_SIZE)
    pages = max(1, math.ceil(total / AUDIT_PAGE_SIZE))
    if page > pages:
        page = pages
        records, total = _store().audit_log(page, AUDIT_PAGE_SIZE)
    return render_template(
        "admin/audit_log.html",
        records=records,
        page=page,
        pages=pages,
        total=total,
    )


@web.get("/health")
def health():
    return {"status": "ok", "service": "anpr-web-demo"}


@web.get("/about")
def about_redirect():
    return redirect(url_for("web.architecture"))


def _checkpoint_action(submitted):
    mode = current_app.config["ANPR_CHECKPOINT_MODE"]
    if mode in {"entry", "exit"}:
        if submitted and submitted != mode:
            return None, "Invalid checkpoint action."
        return mode, None
    if submitted not in {"entry", "exit"}:
        return None, "Select a valid Entry or Exit action before analysis."
    return submitted, None


def _valid_date(value):
    value = value.strip()
    if not value:
        return ""
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return ""
    return value


def _history_filter_values():
    search = request.args.get("q", "").strip()
    status = request.args.get("status", "").strip().lower()
    if status not in {"", "inside", "exited"}:
        status = ""
    date_from = _valid_date(request.args.get("date_from", ""))
    date_to = _valid_date(request.args.get("date_to", ""))
    date_from_utc, date_to_utc = _date_bounds_utc(date_from, date_to)
    return search, status, date_from, date_to, date_from_utc, date_to_utc


def _parse_utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _date_bounds_utc(date_from, date_to):
    zone = current_app.config["ANPR_ZONEINFO"]
    start = (
        datetime.combine(
            datetime.strptime(date_from, "%Y-%m-%d").date(),
            datetime_time.min,
            tzinfo=zone,
        )
        if date_from
        else None
    )
    end = (
        datetime.combine(
            datetime.strptime(date_to, "%Y-%m-%d").date(),
            datetime_time.min,
            tzinfo=zone,
        )
        + timedelta(days=1)
        if date_to
        else None
    )
    return (
        start.astimezone(timezone.utc).isoformat(timespec="seconds") if start else "",
        end.astimezone(timezone.utc).isoformat(timespec="seconds") if end else "",
    )


def _visit_view(record):
    entered = _parse_utc(record["entered_at"])
    exited = _parse_utc(record["exited_at"]) if record["exited_at"] else None
    local_zone = current_app.config["ANPR_ZONEINFO"]
    duration = (exited or datetime.now(timezone.utc)) - entered
    return {
        "plate_number": record["plate_number"],
        "display_name": record["display_name"],
        "category": record["category"],
        "vehicle_record_status": (
            "Deleted vehicle record" if record["vehicle_deleted"] else ""
        ),
        "entry_time": entered.astimezone(local_zone).strftime("%Y-%m-%d %H:%M"),
        "exit_time": (
            exited.astimezone(local_zone).strftime("%Y-%m-%d %H:%M") if exited else "—"
        ),
        "duration": _format_duration(duration.total_seconds()),
        "status": "Exited" if exited else "Inside",
        "delivery_status": _delivery_status_label(
            record["delivery_status"] if "delivery_status" in record.keys() else "",
            (
                record["delivery_error_code"]
                if "delivery_error_code" in record.keys()
                else ""
            ),
        ),
        "delivery_id": (
            record["delivery_id"] if "delivery_id" in record.keys() else None
        ),
        "visit_id": record["visit_id"] if "visit_id" in record.keys() else None,
    }


def _format_duration(seconds):
    total_minutes = max(0, int(seconds // 60))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


def _history_metrics():
    zone = current_app.config["ANPR_ZONEINFO"]
    local_now = datetime.now(zone)
    start = datetime.combine(local_now.date(), datetime_time.min, tzinfo=zone)
    end = start + timedelta(days=1)
    metrics = _store().visit_metrics(
        start.astimezone(timezone.utc).isoformat(timespec="seconds"),
        end.astimezone(timezone.utc).isoformat(timespec="seconds"),
    )
    metrics["average_duration"] = (
        _format_duration(metrics["average_seconds"])
        if metrics["average_seconds"] is not None
        else "—"
    )
    return metrics


def _occupancy_summary():
    inside = _store().occupancy_count()
    capacity = current_app.config["ANPR_PARKING_CAPACITY"]
    if capacity is None:
        return {
            "inside": inside,
            "capacity": None,
            "available": None,
            "percentage": None,
        }
    return {
        "inside": inside,
        "capacity": capacity,
        "available": max(0, capacity - inside),
        "percentage": min(100, round((inside / capacity) * 100)),
    }


def _deliver_exit_summary(visit_id):
    visit = _store().visit_for_notification(visit_id)
    if visit is None or not visit["exited_at"]:
        return None
    email = visit["email"]
    masked = mask_email(email)
    if not visit["email_notifications_enabled"] or not email or not visit["is_active"]:
        _store().create_delivery(
            visit_id,
            visit["vehicle_id"],
            masked,
            "skipped",
            "preference_disabled",
        )
        return "Email notification disabled"
    if not current_app.config["ANPR_EMAIL_READY"]:
        _store().create_delivery(
            visit_id,
            visit["vehicle_id"],
            masked,
            "skipped",
            "not_configured",
        )
        return "Email notification disabled"
    delivery = _store().create_delivery(
        visit_id, visit["vehicle_id"], masked, "pending"
    )
    if delivery["status"] == "sent":
        return "Visit summary email sent"
    return (
        "Visit summary email sent"
        if _attempt_delivery(_store().delivery_context(delivery["id"])) == "sent"
        else "Visit recorded, but email delivery failed"
    )


def _attempt_delivery(context):
    if (
        context is None
        or context["status"] == "sent"
        or not context["email"]
        or not context["email_notifications_enabled"]
        or not context["is_active"]
        or not current_app.config["ANPR_EMAIL_READY"]
    ):
        return "skipped"
    entered = parse_utc(context["entered_at"])
    exited = parse_utc(context["exited_at"])
    summary = VisitSummary(
        plate=context["plate_number"],
        entered_at=entered,
        exited_at=exited,
        duration=_format_duration((exited - entered).total_seconds()),
    )
    try:
        current_app.extensions["anpr_mail"].send_visit_summary(
            context["email"], summary
        )
    except Exception:
        _store().set_delivery_result(context["id"], "failed", "delivery_failed")
        return "failed"
    _store().set_delivery_result(context["id"], "sent")
    return "sent"


def _delivery_status_label(status, error_code=""):
    if status == "skipped" and error_code == "not_configured":
        return "Not configured"
    return {
        "sent": "Sent",
        "failed": "Failed",
        "skipped": "Disabled",
        "pending": "Not configured",
        "": "Not configured",
    }.get(status, "Not configured")


def _csv_safe(value):
    text = str(value)
    return f"'{text}" if text.startswith(("=", "+", "-", "@")) else text
