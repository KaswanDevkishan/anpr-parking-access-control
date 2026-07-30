"""SQLite storage for the Flask access-decision prototype."""

import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from matcher import normalise_plate

CATEGORIES = ("student", "staff", "visitor", "contractor")
SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vehicles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    category TEXT NOT NULL CHECK (
        category IN ('student', 'staff', 'visitor', 'contractor')
    ),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    is_deleted INTEGER NOT NULL DEFAULT 0 CHECK (is_deleted IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_vehicles_plate_number
    ON vehicles (plate_number);
CREATE INDEX IF NOT EXISTS idx_vehicles_display_name
    ON vehicles (display_name);
CREATE INDEX IF NOT EXISTS idx_vehicles_category_active
    ON vehicles (category, is_active);
CREATE TABLE IF NOT EXISTS access_visits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id INTEGER,
    plate_number_snapshot TEXT NOT NULL,
    display_name_snapshot TEXT NOT NULL,
    category_snapshot TEXT NOT NULL,
    entered_at TEXT NOT NULL,
    exited_at TEXT,
    entry_source TEXT NOT NULL,
    exit_source TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (vehicle_id) REFERENCES vehicles (id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_access_visits_entered_at
    ON access_visits (entered_at DESC);
CREATE INDEX IF NOT EXISTS idx_access_visits_exited_at
    ON access_visits (exited_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_access_visits_open_vehicle
    ON access_visits (vehicle_id) WHERE exited_at IS NULL;
CREATE TABLE IF NOT EXISTS notification_deliveries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    visit_id INTEGER NOT NULL,
    vehicle_id INTEGER,
    recipient_email_masked TEXT NOT NULL DEFAULT '',
    notification_type TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'sent', 'failed', 'skipped')),
    attempted_at TEXT,
    sent_at TEXT,
    error_code TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (visit_id) REFERENCES access_visits (id) ON DELETE RESTRICT,
    FOREIGN KEY (vehicle_id) REFERENCES vehicles (id) ON DELETE SET NULL,
    UNIQUE (visit_id, notification_type)
);
CREATE INDEX IF NOT EXISTS idx_notification_deliveries_status
    ON notification_deliveries (status, created_at DESC);
CREATE TABLE IF NOT EXISTS admin_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_reference TEXT NOT NULL,
    summary TEXT NOT NULL,
    occurred_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_admin_audit_log_occurred
    ON admin_audit_log (occurred_at DESC, id DESC);
"""


class DuplicatePlate(ValueError):
    """Raised when a normalized plate already exists."""


class VehicleCurrentlyInside(ValueError):
    """Raised when permanent deletion is attempted during an open visit."""


@dataclass(frozen=True)
class CheckpointResult:
    status: str
    visit_id: int | None = None
    vehicle_id: int | None = None


class VehicleStore:
    """Small repository that opens short-lived SQLite connections."""

    def __init__(self, path):
        self.path = Path(path)

    def initialize(self, seed_csv=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            had_vehicle_table = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'vehicles'
                """
            ).fetchone()
            connection.executescript(SCHEMA)
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(vehicles)").fetchall()
            }
            if "is_deleted" not in columns:
                connection.execute(
                    """
                    ALTER TABLE vehicles
                    ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0
                    """
                )
            additions = {
                "email": "TEXT",
                "email_notifications_enabled": (
                    "INTEGER NOT NULL DEFAULT 0 "
                    "CHECK (email_notifications_enabled IN (0, 1))"
                ),
                "email_verified_at": "TEXT",
            }
            for name, definition in additions.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE vehicles ADD COLUMN {name} {definition}"
                    )
            connection.commit()
            self._migrate_historical_records(connection)
            connection.execute(
                """
                INSERT OR IGNORE INTO schema_migrations (version, applied_at)
                VALUES (1, ?)
                """,
                (_utc_now(),),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO schema_migrations (version, applied_at)
                VALUES (4, ?)
                """,
                (_utc_now(),),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO schema_migrations (version, applied_at)
                VALUES (3, ?)
                """,
                (_utc_now(),),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO schema_migrations (version, applied_at)
                VALUES (2, ?)
                """,
                (_utc_now(),),
            )
            count = connection.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]
            if count == 0 and seed_csv and not had_vehicle_table:
                self._seed(connection, seed_csv)

    def list_public(self, search="", category=""):
        clauses = []
        parameters = []
        if search:
            clauses.append("(plate_number LIKE ? OR display_name LIKE ?)")
            plate_term = f"%{normalise_plate(search)}%"
            name_term = f"%{search.strip()}%"
            parameters.extend((plate_term, name_term))
        if category in CATEGORIES:
            clauses.append("category = ?")
            parameters.append(category)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            "SELECT plate_number, display_name, category, is_active "
            f"FROM vehicles{where} ORDER BY plate_number"
        )
        with self._connect() as connection:
            return connection.execute(query, parameters).fetchall()

    def list_recent(self, limit=5):
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT * FROM vehicles
                ORDER BY created_at DESC, id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()

    def list_all(self):
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM vehicles ORDER BY plate_number"
            ).fetchall()

    def counts(self):
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) AS active,
                       SUM(CASE WHEN is_active = 0 THEN 1 ELSE 0 END) AS inactive
                FROM vehicles
                """
            ).fetchone()
        return {
            "total": row["total"],
            "active": row["active"] or 0,
            "inactive": row["inactive"] or 0,
        }

    def get(self, vehicle_id):
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM vehicles WHERE id = ?",
                (vehicle_id,),
            ).fetchone()

    def get_active_by_plate(self, plate_number):
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT * FROM vehicles
                WHERE plate_number = ? AND is_active = 1
                """,
                (normalise_plate(plate_number),),
            ).fetchone()

    def create(
        self,
        plate_number,
        display_name,
        category,
        is_active=True,
        email=None,
        email_notifications_enabled=False,
        email_verified_at=None,
    ):
        normalized = normalise_plate(plate_number)
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO vehicles
                        (plate_number, display_name, category, is_active, email,
                         email_notifications_enabled, email_verified_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized,
                        display_name.strip(),
                        category,
                        int(is_active),
                        email,
                        int(email_notifications_enabled),
                        email_verified_at,
                    ),
                )
                return cursor.lastrowid
        except sqlite3.IntegrityError as exc:
            if "plate_number" in str(exc):
                raise DuplicatePlate(normalized) from exc
            raise

    def update_vehicle(
        self,
        vehicle_id,
        *,
        plate_number,
        display_name,
        category,
        is_active,
        email,
        email_notifications_enabled,
    ):
        normalized = normalise_plate(plate_number)
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE vehicles
                    SET plate_number = ?, display_name = ?, category = ?,
                        is_active = ?, email = ?,
                        email_notifications_enabled = ?,
                        email_verified_at = CASE
                            WHEN email IS NOT ? THEN NULL ELSE email_verified_at END,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        normalized,
                        display_name.strip(),
                        category,
                        int(is_active),
                        email,
                        int(email_notifications_enabled),
                        email,
                        _utc_now(),
                        vehicle_id,
                    ),
                )
                return cursor.rowcount == 1
        except sqlite3.IntegrityError as exc:
            if "plate_number" in str(exc):
                raise DuplicatePlate(normalized) from exc
            raise

    def set_active(self, vehicle_id, is_active):
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE vehicles
                SET is_active = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (int(is_active), vehicle_id),
            )
            return cursor.rowcount == 1

    def delete(self, vehicle_id):
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            vehicle = connection.execute(
                "SELECT plate_number, category FROM vehicles WHERE id = ?",
                (vehicle_id,),
            ).fetchone()
            if vehicle is None:
                return False
            open_visit = connection.execute(
                """
                SELECT 1 FROM access_visits
                WHERE vehicle_id = ? AND exited_at IS NULL
                """,
                (vehicle_id,),
            ).fetchone()
            if open_visit:
                raise VehicleCurrentlyInside(vehicle["plate_number"])
            connection.execute(
                """
                INSERT INTO admin_audit_log
                    (action, target_type, target_reference, summary, occurred_at)
                VALUES ('vehicle_deleted', 'vehicle', ?, ?, ?)
                """,
                (
                    _masked_plate(vehicle["plate_number"]),
                    f"Vehicle permanently deleted ({vehicle['category']}).",
                    _utc_now(),
                ),
            )
            cursor = connection.execute(
                "DELETE FROM vehicles WHERE id = ?",
                (vehicle_id,),
            )
            return cursor.rowcount == 1

    def apply_checkpoint(self, plate_number, action, source="web-demo", now=None):
        """Atomically record an entry or exit for one exact active vehicle."""
        return self.apply_checkpoint_detail(plate_number, action, source, now).status

    def apply_checkpoint_detail(
        self, plate_number, action, source="web-demo", now=None
    ):
        """Atomically apply a checkpoint and return identifiers after commit."""
        if action not in {"entry", "exit"}:
            raise ValueError("Checkpoint action must be entry or exit.")
        timestamp = now or _utc_now()
        normalized = normalise_plate(plate_number)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            vehicle = connection.execute(
                """
                SELECT id, plate_number, display_name, category FROM vehicles
                WHERE plate_number = ? AND is_active = 1
                """,
                (normalized,),
            ).fetchone()
            if vehicle is None:
                return CheckpointResult("denied")
            open_visit = connection.execute(
                """
                SELECT id FROM access_visits
                WHERE vehicle_id = ? AND exited_at IS NULL
                LIMIT 1
                """,
                (vehicle["id"],),
            ).fetchone()
            if action == "entry":
                if open_visit:
                    return CheckpointResult(
                        "already_inside", open_visit["id"], vehicle["id"]
                    )
                cursor = connection.execute(
                    """
                    INSERT INTO access_visits
                        (vehicle_id, plate_number_snapshot, display_name_snapshot,
                         category_snapshot, entered_at, entry_source, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        vehicle["id"],
                        vehicle["plate_number"],
                        vehicle["display_name"],
                        vehicle["category"],
                        timestamp,
                        source,
                        timestamp,
                    ),
                )
                return CheckpointResult(
                    "entry_recorded", cursor.lastrowid, vehicle["id"]
                )
            if not open_visit:
                return CheckpointResult("no_active_entry", vehicle_id=vehicle["id"])
            connection.execute(
                """
                UPDATE access_visits
                SET exited_at = ?, exit_source = ?
                WHERE id = ? AND exited_at IS NULL
                """,
                (timestamp, source, open_visit["id"]),
            )
            return CheckpointResult("exit_recorded", open_visit["id"], vehicle["id"])

    def visit_for_notification(self, visit_id):
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT a.id AS visit_id, a.entered_at, a.exited_at,
                       v.id AS vehicle_id, v.plate_number, v.display_name,
                       v.email, v.email_notifications_enabled, v.is_active
                FROM access_visits a JOIN vehicles v ON v.id = a.vehicle_id
                WHERE a.id = ?
                """,
                (visit_id,),
            ).fetchone()

    def create_delivery(
        self, visit_id, vehicle_id, recipient_email_masked, status, error_code=None
    ):
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO notification_deliveries
                    (visit_id, vehicle_id, recipient_email_masked,
                     notification_type, status, error_code, created_at)
                VALUES (?, ?, ?, 'exit_summary', ?, ?, ?)
                """,
                (
                    visit_id,
                    vehicle_id,
                    recipient_email_masked,
                    status,
                    error_code,
                    now,
                ),
            )
            return connection.execute(
                """
                SELECT * FROM notification_deliveries
                WHERE visit_id = ? AND notification_type = 'exit_summary'
                """,
                (visit_id,),
            ).fetchone()

    def get_delivery(self, delivery_id):
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM notification_deliveries WHERE id = ?",
                (delivery_id,),
            ).fetchone()

    def delivery_for_visit(self, visit_id):
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT * FROM notification_deliveries
                WHERE visit_id = ? AND notification_type = 'exit_summary'
                """,
                (visit_id,),
            ).fetchone()

    def count_deliveries(self):
        with self._connect() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM notification_deliveries"
            ).fetchone()[0]

    def set_delivery_result(self, delivery_id, status, error_code=None):
        now = _utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE notification_deliveries
                SET status = ?, attempted_at = ?,
                    sent_at = CASE WHEN ? = 'sent' THEN ? ELSE sent_at END,
                    error_code = ?
                WHERE id = ? AND status != 'sent'
                """,
                (status, now, status, now, error_code, delivery_id),
            )
            return cursor.rowcount == 1

    def delivery_context(self, delivery_id):
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT d.*, a.entered_at, a.exited_at, v.plate_number, v.email,
                       v.email_notifications_enabled, v.is_active
                FROM notification_deliveries d
                JOIN access_visits a ON a.id = d.visit_id
                JOIN vehicles v ON v.id = d.vehicle_id
                WHERE d.id = ?
                """,
                (delivery_id,),
            ).fetchone()

    def occupancy_count(self):
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT COUNT(*) FROM access_visits
                WHERE exited_at IS NULL AND vehicle_id IS NOT NULL
                """
            ).fetchone()[0]

    def count_visits(self):
        with self._connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM access_visits").fetchone()[
                0
            ]

    def history(
        self, search="", status="", date_from="", date_to="", page=1, per_page=10
    ):
        clauses, parameters = [], []
        if search:
            clauses.append(
                "(COALESCE(v.plate_number, a.plate_number_snapshot) LIKE ? "
                "OR COALESCE(v.display_name, a.display_name_snapshot) LIKE ?)"
            )
            parameters.extend((f"%{normalise_plate(search)}%", f"%{search.strip()}%"))
        if status == "inside":
            clauses.append("a.exited_at IS NULL")
        elif status == "exited":
            clauses.append("a.exited_at IS NOT NULL")
        if date_from:
            clauses.append("a.entered_at >= ?")
            parameters.append(date_from)
        if date_to:
            clauses.append("a.entered_at < ?")
            parameters.append(date_to)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        joins = """
            FROM access_visits a
            LEFT JOIN vehicles v ON v.id = a.vehicle_id
            LEFT JOIN notification_deliveries d
              ON d.visit_id = a.id AND d.notification_type = 'exit_summary'
        """
        base = f"{joins}{where}"
        with self._connect() as connection:
            total = connection.execute(f"SELECT COUNT(*){base}", parameters).fetchone()[
                0
            ]
            records = connection.execute(
                """
                SELECT COALESCE(v.plate_number, a.plate_number_snapshot)
                           AS plate_number,
                       COALESCE(v.display_name, a.display_name_snapshot)
                           AS display_name,
                       COALESCE(v.category, a.category_snapshot) AS category,
                       CASE WHEN v.id IS NULL THEN 1 ELSE 0 END AS vehicle_deleted,
                       a.id AS visit_id, a.entered_at, a.exited_at,
                       d.id AS delivery_id,
                       COALESCE(d.status, '') AS delivery_status,
                       COALESCE(d.error_code, '') AS delivery_error_code
                """
                + base
                + " ORDER BY a.entered_at DESC, a.id DESC LIMIT ? OFFSET ?",
                (*parameters, per_page, (page - 1) * per_page),
            ).fetchall()
        return records, total

    def history_export(self, search="", status="", date_from="", date_to=""):
        records, _ = self.history(
            search, status, date_from, date_to, page=1, per_page=1_000_000
        )
        return records

    def audit(self, action, target_type, target_reference, summary):
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO admin_audit_log
                    (action, target_type, target_reference, summary, occurred_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (action, target_type, target_reference, summary, _utc_now()),
            )

    def audit_log(self, page=1, per_page=20):
        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM admin_audit_log"
            ).fetchone()[0]
            rows = connection.execute(
                """
                SELECT * FROM admin_audit_log
                ORDER BY occurred_at DESC, id DESC LIMIT ? OFFSET ?
                """,
                (per_page, (page - 1) * per_page),
            ).fetchall()
        return rows, total

    def visit_metrics(self, local_day_start_utc, local_day_end_utc):
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                  SUM(CASE WHEN exited_at IS NULL AND vehicle_id IS NOT NULL
                      THEN 1 ELSE 0 END) AS inside,
                  SUM(CASE WHEN entered_at >= ? AND entered_at < ? THEN 1 ELSE 0 END)
                    AS entries_today,
                  SUM(CASE WHEN exited_at >= ? AND exited_at < ? THEN 1 ELSE 0 END)
                    AS exits_today,
                  AVG(CASE WHEN exited_at IS NOT NULL
                    THEN (julianday(exited_at) - julianday(entered_at)) * 86400
                    END) AS average_seconds
                FROM access_visits
                """,
                (
                    local_day_start_utc,
                    local_day_end_utc,
                    local_day_start_utc,
                    local_day_end_utc,
                ),
            ).fetchone()
        return {
            "inside": row["inside"] or 0,
            "entries_today": row["entries_today"] or 0,
            "exits_today": row["exits_today"] or 0,
            "average_seconds": row["average_seconds"],
        }

    def active_match_database(self):
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT plate_number, display_name, category
                FROM vehicles WHERE is_active = 1
                """
            ).fetchall()
        return {
            row["plate_number"]: {
                "name": row["display_name"],
                "id": "",
                "type": row["category"],
            }
            for row in rows
        }

    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @staticmethod
    def _migrate_historical_records(connection):
        """Idempotently snapshot visits and install deletion-safe foreign keys."""
        visit_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(access_visits)")
        }
        snapshot_definitions = {
            "plate_number_snapshot": "TEXT",
            "display_name_snapshot": "TEXT",
            "category_snapshot": "TEXT",
        }
        for name, definition in snapshot_definitions.items():
            if name not in visit_columns:
                connection.execute(
                    f"ALTER TABLE access_visits ADD COLUMN {name} {definition}"
                )
        connection.execute(
            """
            UPDATE access_visits
            SET plate_number_snapshot = COALESCE(
                    plate_number_snapshot,
                    (SELECT plate_number FROM vehicles
                     WHERE vehicles.id = access_visits.vehicle_id),
                    'UNKNOWN'
                ),
                display_name_snapshot = COALESCE(
                    display_name_snapshot,
                    (SELECT display_name FROM vehicles
                     WHERE vehicles.id = access_visits.vehicle_id),
                    'Unknown vehicle'
                ),
                category_snapshot = COALESCE(
                    category_snapshot,
                    (SELECT category FROM vehicles
                     WHERE vehicles.id = access_visits.vehicle_id),
                    'unknown'
                )
            """
        )
        connection.commit()
        visit_fk = connection.execute(
            "PRAGMA foreign_key_list(access_visits)"
        ).fetchall()
        delivery_fk = connection.execute(
            "PRAGMA foreign_key_list(notification_deliveries)"
        ).fetchall()
        visits_safe = any(
            row["table"] == "vehicles" and row["on_delete"] == "SET NULL"
            for row in visit_fk
        )
        deliveries_safe = any(
            row["table"] == "vehicles" and row["on_delete"] == "SET NULL"
            for row in delivery_fk
        )
        if not (visits_safe and deliveries_safe):
            VehicleStore._rebuild_history_tables(connection)
        # Retire legacy tombstones only after snapshots and SET NULL are in place.
        connection.execute("DELETE FROM vehicles WHERE is_deleted = 1")

    @staticmethod
    def _rebuild_history_tables(connection):
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.executescript(
            """
            BEGIN IMMEDIATE;
            ALTER TABLE access_visits RENAME TO access_visits_legacy;
            ALTER TABLE notification_deliveries
                RENAME TO notification_deliveries_legacy;
            CREATE TABLE access_visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id INTEGER,
                plate_number_snapshot TEXT NOT NULL,
                display_name_snapshot TEXT NOT NULL,
                category_snapshot TEXT NOT NULL,
                entered_at TEXT NOT NULL,
                exited_at TEXT,
                entry_source TEXT NOT NULL,
                exit_source TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (vehicle_id) REFERENCES vehicles (id)
                    ON DELETE SET NULL
            );
            INSERT INTO access_visits
            SELECT id, vehicle_id, plate_number_snapshot, display_name_snapshot,
                   category_snapshot, entered_at, exited_at, entry_source,
                   exit_source, created_at
            FROM access_visits_legacy;
            CREATE TABLE notification_deliveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                visit_id INTEGER NOT NULL,
                vehicle_id INTEGER,
                recipient_email_masked TEXT NOT NULL DEFAULT '',
                notification_type TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN ('pending', 'sent', 'failed', 'skipped')
                ),
                attempted_at TEXT,
                sent_at TEXT,
                error_code TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (visit_id) REFERENCES access_visits (id)
                    ON DELETE RESTRICT,
                FOREIGN KEY (vehicle_id) REFERENCES vehicles (id)
                    ON DELETE SET NULL,
                UNIQUE (visit_id, notification_type)
            );
            INSERT INTO notification_deliveries
            SELECT id, visit_id, vehicle_id, recipient_email_masked,
                   notification_type, status, attempted_at, sent_at, error_code,
                   created_at
            FROM notification_deliveries_legacy;
            DROP TABLE notification_deliveries_legacy;
            DROP TABLE access_visits_legacy;
            CREATE INDEX idx_access_visits_entered_at
                ON access_visits (entered_at DESC);
            CREATE INDEX idx_access_visits_exited_at
                ON access_visits (exited_at);
            CREATE UNIQUE INDEX idx_access_visits_open_vehicle
                ON access_visits (vehicle_id) WHERE exited_at IS NULL;
            CREATE INDEX idx_notification_deliveries_status
                ON notification_deliveries (status, created_at DESC);
            COMMIT;
            """
        )
        connection.execute("PRAGMA foreign_keys = ON")

    @staticmethod
    def _seed(connection, seed_csv):
        seed_path = Path(seed_csv)
        if not seed_path.is_file():
            return
        with seed_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                plate = normalise_plate(row.get("plate_number"))
                name = (row.get("name") or "").strip()
                category = (row.get("type") or "").strip().lower()
                if plate and name and category in CATEGORIES:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO vehicles
                            (plate_number, display_name, category, is_active)
                        VALUES (?, ?, ?, 1)
                        """,
                        (plate, name, category),
                    )


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _masked_plate(plate_number):
    plate = normalise_plate(plate_number)
    if len(plate) <= 2:
        return "*" * len(plate)
    return f"{plate[:1]}{'*' * (len(plate) - 2)}{plate[-1:]}"
