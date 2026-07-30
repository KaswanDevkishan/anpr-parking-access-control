import sqlite3

import pytest

from anpr_web.database import DuplicatePlate, VehicleCurrentlyInside, VehicleStore
from anpr_web.vehicles import save_vehicle, vehicle_form_values


def seed_csv(path, plate="SEED1"):
    path.write_text(
        f"plate_number,name,id,type\n{plate},Fictional Owner,DEMO-ONLY,visitor\n",
        encoding="utf-8",
    )


def test_seed_loads_only_when_database_is_empty(tmp_path):
    csv_path = tmp_path / "seed.csv"
    database_path = tmp_path / "vehicles.sqlite3"
    seed_csv(csv_path)
    store = VehicleStore(database_path)

    store.initialize(csv_path)
    assert [row["plate_number"] for row in store.list_all()] == ["SEED1"]

    store.create("KEEP2", "Existing Fiction", "staff")
    seed_csv(csv_path, "REPLACE3")
    store.initialize(csv_path)

    plates = {row["plate_number"] for row in store.list_all()}
    assert plates == {"SEED1", "KEEP2"}


def test_deleted_last_vehicle_is_not_reseeded_on_restart(tmp_path):
    csv_path = tmp_path / "seed.csv"
    database_path = tmp_path / "vehicles.sqlite3"
    seed_csv(csv_path)
    store = VehicleStore(database_path)
    store.initialize(csv_path)
    assert store.delete(store.list_all()[0]["id"])

    store.initialize(csv_path)

    assert store.list_all() == []


def test_entry_exit_is_transactional_and_idempotent(tmp_path):
    store = VehicleStore(tmp_path / "visits.sqlite3")
    store.initialize()
    store.create("VISIT1", "Fictional Visitor", "visitor")

    assert store.apply_checkpoint("VISIT1", "entry") == "entry_recorded"
    assert store.apply_checkpoint("VISIT1", "entry") == "already_inside"
    assert store.count_visits() == 1
    assert store.apply_checkpoint("VISIT1", "exit") == "exit_recorded"
    assert store.apply_checkpoint("VISIT1", "exit") == "no_active_entry"
    assert store.count_visits() == 1


def test_denied_vehicles_never_change_visits(tmp_path):
    store = VehicleStore(tmp_path / "denied.sqlite3")
    store.initialize()
    inactive = store.create("OFF1", "Inactive Fiction", "staff", False)

    assert store.apply_checkpoint("UNKNOWN", "entry") == "denied"
    assert store.apply_checkpoint("OFF1", "entry") == "denied"
    store.set_active(inactive, True)
    store.delete(inactive)
    assert store.apply_checkpoint("OFF1", "entry") == "denied"
    assert store.count_visits() == 0


def test_permanent_deletion_retains_snapshots_and_allows_new_vehicle(tmp_path):
    store = VehicleStore(tmp_path / "history.sqlite3")
    store.initialize()
    vehicle_id = store.create("KEEP1", "Retained Fiction", "contractor")
    store.apply_checkpoint("KEEP1", "entry")
    store.apply_checkpoint("KEEP1", "exit")

    assert store.delete(vehicle_id)
    assert store.get(vehicle_id) is None
    records, total = store.history()
    assert total == 1
    assert records[0]["plate_number"] == "KEEP1"
    assert records[0]["display_name"] == "Retained Fiction"
    assert records[0]["category"] == "contractor"
    assert records[0]["vehicle_deleted"] == 1
    new_id = store.create(" keep1 ", "New Fiction", "student")
    assert new_id != vehicle_id
    assert store.history()[0][0]["display_name"] == "Retained Fiction"
    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute("SELECT vehicle_id FROM access_visits").fetchone()[0]
            is None
        )


def test_duplicate_and_open_visit_deletion_rules(tmp_path):
    store = VehicleStore(tmp_path / "delete-rules.sqlite3")
    store.initialize()
    vehicle_id = store.create("RULE1", "Rule Fiction", "visitor")
    with pytest.raises(DuplicatePlate):
        store.create(" rule1 ", "Duplicate Fiction", "staff")
    store.apply_checkpoint("RULE1", "entry")
    with pytest.raises(VehicleCurrentlyInside):
        store.delete(vehicle_id)
    assert store.get(vehicle_id) is not None
    store.apply_checkpoint("RULE1", "exit")
    assert store.delete(vehicle_id)


def test_notification_delivery_survives_vehicle_deletion(tmp_path):
    store = VehicleStore(tmp_path / "delivery-history.sqlite3")
    store.initialize()
    vehicle_id = store.create("MAILH1", "Mail History", "visitor")
    checkpoint = store.apply_checkpoint_detail("MAILH1", "entry")
    store.apply_checkpoint("MAILH1", "exit")
    delivery = store.create_delivery(
        checkpoint.visit_id, vehicle_id, "m***@example.test", "sent"
    )
    assert store.delete(vehicle_id)
    retained = store.get_delivery(delivery["id"])
    assert retained["visit_id"] == checkpoint.visit_id
    assert retained["vehicle_id"] is None
    assert retained["recipient_email_masked"] == "m***@example.test"


def test_legacy_history_migration_is_idempotent_and_backfills_snapshots(tmp_path):
    path = tmp_path / "legacy-history.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                category TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                is_deleted INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE access_visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id INTEGER NOT NULL,
                entered_at TEXT NOT NULL,
                exited_at TEXT,
                entry_source TEXT NOT NULL,
                exit_source TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (vehicle_id) REFERENCES vehicles (id)
                    ON DELETE RESTRICT
            );
            CREATE TABLE notification_deliveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                visit_id INTEGER NOT NULL,
                vehicle_id INTEGER NOT NULL,
                recipient_email_masked TEXT NOT NULL DEFAULT '',
                notification_type TEXT NOT NULL,
                status TEXT NOT NULL,
                attempted_at TEXT,
                sent_at TEXT,
                error_code TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (visit_id) REFERENCES access_visits (id)
                    ON DELETE RESTRICT,
                FOREIGN KEY (vehicle_id) REFERENCES vehicles (id)
                    ON DELETE RESTRICT,
                UNIQUE (visit_id, notification_type)
            );
            INSERT INTO vehicles
                (plate_number, display_name, category, is_deleted)
            VALUES ('OLD1', 'Old Fiction', 'staff', 1);
            INSERT INTO access_visits
                (vehicle_id, entered_at, exited_at, entry_source, exit_source,
                 created_at)
            VALUES (1, '2026-01-01T00:00:00+00:00',
                    '2026-01-01T01:00:00+00:00', 'legacy', 'legacy',
                    '2026-01-01T00:00:00+00:00');
            INSERT INTO notification_deliveries
                (visit_id, vehicle_id, recipient_email_masked,
                 notification_type, status, created_at)
            VALUES (1, 1, 'o***@example.test', 'exit_summary', 'sent',
                    '2026-01-01T01:00:00+00:00');
            """
        )
    store = VehicleStore(path)
    store.initialize()
    store.initialize()
    record = store.history()[0][0]
    assert record["plate_number"] == "OLD1"
    assert record["display_name"] == "Old Fiction"
    assert record["vehicle_deleted"] == 1
    assert store.get_delivery(1)["vehicle_id"] is None
    assert store.create("OLD1", "New Fiction", "visitor") > 1
    with store._connect() as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        visit_fk = connection.execute(
            "PRAGMA foreign_key_list(access_visits)"
        ).fetchall()
        assert any(row["on_delete"] == "SET NULL" for row in visit_fk)


def test_email_columns_migrate_existing_vehicle_database(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                category TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            INSERT INTO vehicles (plate_number, display_name, category)
            VALUES ('LEGACY1', 'Legacy Example', 'visitor')
            """
        )
    store = VehicleStore(path)
    store.initialize()
    vehicle = store.list_all()[0]
    assert vehicle["email"] is None
    assert vehicle["email_notifications_enabled"] == 0
    assert vehicle["email_verified_at"] is None


def test_shared_vehicle_email_validation_and_normalization(tmp_path):
    store = VehicleStore(tmp_path / "email.sqlite3")
    store.initialize()
    values = vehicle_form_values(
        {
            "plate_number": "MAIL1",
            "display_name": "Mail Example",
            "category": "visitor",
            "is_active": "1",
            "email": " Person@EXAMPLE.COM ",
            "email_notifications_enabled": "1",
        }
    )
    vehicle_id, errors = save_vehicle(store, values)
    assert errors == []
    assert store.get(vehicle_id)["email"] == "Person@example.com"
