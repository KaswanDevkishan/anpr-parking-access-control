"""Session authentication, CSRF, and login throttling helpers."""

import hmac
import secrets
import time
from functools import wraps
from urllib.parse import urljoin, urlparse

from flask import (
    abort,
    current_app,
    redirect,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash

CSRF_SESSION_KEY = "_csrf_token"
LOGIN_SESSION_KEY = "admin_authenticated"


def csrf_token():
    token = session.get(CSRF_SESSION_KEY)
    if token is None:
        token = secrets.token_urlsafe(32)
        session[CSRF_SESSION_KEY] = token
    return token


def protect_post_requests():
    if request.method != "POST":
        return None
    submitted = request.form.get("csrf_token", "")
    expected = session.get(CSRF_SESSION_KEY, "")
    if not expected or not hmac.compare_digest(submitted, expected):
        abort(400, description="Invalid or missing CSRF token.")
    return None


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get(LOGIN_SESSION_KEY):
            return redirect(url_for("web.admin_login", next=request.full_path))
        return view(*args, **kwargs)

    return wrapped


def is_safe_redirect(target):
    if not target:
        return False
    host = urlparse(request.host_url)
    destination = urlparse(urljoin(request.host_url, target))
    return destination.scheme in {"http", "https"} and host.netloc == destination.netloc


def login_locked(client_key):
    attempts = current_app.extensions["anpr_login_attempts"]
    record = attempts.get(client_key)
    if not record:
        return False
    now = time.monotonic()
    if record["locked_until"] > now:
        return True
    if record["locked_until"]:
        attempts.pop(client_key, None)
    return False


def record_login_failure(client_key):
    attempts = current_app.extensions["anpr_login_attempts"]
    now = time.monotonic()
    record = attempts.get(client_key, {"count": 0, "locked_until": 0})
    record["count"] += 1
    if record["count"] >= current_app.config["ADMIN_LOGIN_MAX_ATTEMPTS"]:
        record["locked_until"] = now + current_app.config["ADMIN_LOCKOUT_SECONDS"]
    attempts[client_key] = record


def authenticate_admin(username, password):
    expected_username = current_app.config["ADMIN_USERNAME"]
    password_hash = current_app.config["ADMIN_PASSWORD_HASH"]
    username_matches = hmac.compare_digest(username, expected_username)
    password_matches = bool(password_hash) and check_password_hash(
        password_hash, password
    )
    return username_matches and password_matches
