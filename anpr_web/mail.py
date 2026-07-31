"""Privacy-conscious exit-summary email delivery."""

import json
import smtplib
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from html import escape


class MailDeliveryError(RuntimeError):
    """Safe mail failure without provider details."""


@dataclass(frozen=True)
class VisitSummary:
    plate: str
    entered_at: datetime
    exited_at: datetime
    duration: str


class SMTPBackend:
    def __init__(self, config):
        self.config = config

    def send(self, message):
        try:
            with smtplib.SMTP(
                self.config["ANPR_SMTP_HOST"],
                self.config["ANPR_SMTP_PORT"],
                timeout=self.config["ANPR_EMAIL_TIMEOUT_SECONDS"],
            ) as client:
                if self.config["ANPR_SMTP_USE_TLS"]:
                    client.starttls()
                username = self.config["ANPR_SMTP_USERNAME"]
                if username:
                    client.login(username, self.config["ANPR_SMTP_PASSWORD"])
                client.send_message(message)
        except (OSError, smtplib.SMTPException) as exc:
            raise MailDeliveryError("smtp_delivery_failed") from exc


class HTTPAPIBackend:
    """Provider-neutral JSON email adapter using outbound HTTPS."""

    def __init__(self, config, opener=urllib.request.urlopen):
        self.config = config
        self.opener = opener

    def send(self, message):
        html_part = message.get_body(preferencelist=("html",))
        text_part = message.get_body(preferencelist=("plain",))
        payload = {
            "from": str(message["From"]),
            "to": str(message["To"]),
            "subject": str(message["Subject"]),
            "text": text_part.get_content() if text_part else "",
            "html": html_part.get_content() if html_part else "",
        }
        request = urllib.request.Request(
            self.config["ANPR_EMAIL_API_URL"],
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config['ANPR_EMAIL_API_KEY']}",
                "Content-Type": "application/json",
                "User-Agent": "anpr-access-decision-prototype",
            },
            method="POST",
        )
        try:
            with self.opener(
                request, timeout=self.config["ANPR_EMAIL_TIMEOUT_SECONDS"]
            ) as response:
                if not 200 <= response.status < 300:
                    raise MailDeliveryError("http_api_delivery_failed")
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise MailDeliveryError("http_api_delivery_failed") from exc


class DryRunBackend:
    """In-memory backend suitable for tests and local previews."""

    def __init__(self):
        self.previews = []

    def send(self, message):
        self.previews.append(
            {
                "to": mask_email(str(message["To"])),
                "subject": str(message["Subject"]),
                "text": message.get_body(preferencelist=("plain",)).get_content(),
            }
        )


class MailService:
    def __init__(self, config, backend):
        self.config = config
        self.backend = backend

    def send_visit_summary(self, recipient, summary):
        message = EmailMessage()
        name = self.config["ANPR_EMAIL_FROM_NAME"]
        message["From"] = f"{name} <{self.config['ANPR_EMAIL_FROM']}>"
        message["To"] = recipient
        message["Subject"] = f"{self.config['ANPR_APPLICATION_NAME']} visit completed"
        plate = mask_plate(summary.plate)
        entry = self._display_time(summary.entered_at)
        exit_time = self._display_time(summary.exited_at)
        text = (
            f"{self.config['ANPR_APPLICATION_NAME']}\n\n"
            "Your vehicle visit was completed.\n"
            f"Vehicle: {plate}\nEntry: {entry}\nExit: {exit_time}\n"
            f"Duration: {summary.duration}\n\n"
            "This is an automated visit-summary notification."
        )
        html = (
            '<div style="font-family:system-ui,sans-serif;max-width:600px;margin:auto">'
            f"<h1>{escape(self.config['ANPR_APPLICATION_NAME'])}</h1>"
            "<p>Your vehicle visit was completed.</p><table>"
            f"<tr><th>Vehicle</th><td>{escape(plate)}</td></tr>"
            f"<tr><th>Entry</th><td>{escape(entry)}</td></tr>"
            f"<tr><th>Exit</th><td>{escape(exit_time)}</td></tr>"
            f"<tr><th>Duration</th><td>{escape(summary.duration)}</td></tr>"
            "</table><p>This is an automated visit-summary notification.</p></div>"
        )
        message.set_content(text)
        message.add_alternative(html, subtype="html")
        self.backend.send(message)

    def _display_time(self, value):
        return value.astimezone(self.config["ANPR_ZONEINFO"]).strftime(
            "%Y-%m-%d %H:%M %Z"
        )


def mask_email(value):
    if not value or "@" not in value:
        return ""
    local, domain = value.rsplit("@", 1)
    visible = local[:1]
    return f"{visible}{'*' * max(2, len(local) - 1)}@{domain}"


def mask_plate(value):
    value = value or ""
    visible = value[-3:] if len(value) > 3 else value[-2:]
    return f"{'*' * max(2, len(value) - len(visible))}{visible}"


def parse_utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
