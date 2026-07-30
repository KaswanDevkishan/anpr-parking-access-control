"""Shared validation and persistence for admin-managed vehicle records."""

import re

from matcher import normalise_plate

from .database import CATEGORIES, DuplicatePlate

MAX_PLATE_LENGTH = 20
MAX_NAME_LENGTH = 80
MAX_EMAIL_LENGTH = 254
EMAIL_PATTERN = re.compile(
    r"^[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?"
    r"(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+$",
    re.IGNORECASE,
)


def normalize_email(value):
    value = (value or "").strip()
    if not value:
        return None
    local, separator, domain = value.rpartition("@")
    return f"{local}@{domain.lower()}" if separator else value


def vehicle_form_values(form, existing=None):
    default_active = bool(existing["is_active"]) if existing else True
    return {
        "plate_number": normalise_plate(
            form.get("plate_number", existing["plate_number"] if existing else "")
        ),
        "display_name": form.get(
            "display_name", existing["display_name"] if existing else ""
        ).strip(),
        "email": normalize_email(
            form.get("email", existing["email"] if existing else "")
        ),
        "category": form.get("category", existing["category"] if existing else "")
        .strip()
        .lower(),
        "is_active": (
            form.get("is_active") in {"1", "true", "on"} if form else default_active
        ),
        "email_notifications_enabled": (
            form.get("email_notifications_enabled") in {"1", "true", "on"}
            if form
            else bool(existing and existing["email_notifications_enabled"])
        ),
    }


def validate_vehicle(values):
    errors = []
    if not values["plate_number"]:
        errors.append("Plate number is required.")
    elif len(values["plate_number"]) > MAX_PLATE_LENGTH:
        errors.append(f"Plate number must be {MAX_PLATE_LENGTH} characters or fewer.")
    if not values["display_name"]:
        errors.append("Display owner name is required.")
    elif len(values["display_name"]) > MAX_NAME_LENGTH:
        errors.append(
            f"Display owner name must be {MAX_NAME_LENGTH} characters or fewer."
        )
    email = values["email"]
    if email and (len(email) > MAX_EMAIL_LENGTH or not EMAIL_PATTERN.fullmatch(email)):
        errors.append("Enter a valid email address.")
    if values["email_notifications_enabled"] and not email:
        errors.append("An email address is required to enable visit-summary emails.")
    if values["category"] not in CATEGORIES:
        errors.append("Choose a valid vehicle category.")
    return errors


def save_vehicle(store, values, vehicle_id=None):
    errors = validate_vehicle(values)
    if errors:
        return None, errors
    try:
        if vehicle_id is None:
            return store.create(**values), []
        if not store.update_vehicle(vehicle_id, **values):
            return None, ["Vehicle not found."]
        return vehicle_id, []
    except DuplicatePlate:
        return None, ["That normalized plate number is already registered."]
