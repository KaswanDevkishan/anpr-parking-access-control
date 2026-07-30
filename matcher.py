"""
matcher.py
─────────────────────────────────────────────────────────────
Loads and validates the registered vehicle database, then matches OCR results.
Exact matching is the authorization default. Fuzzy matching is experimental
and rejects ties rather than silently choosing an ambiguous vehicle.
"""

import csv
from pathlib import Path

from Levenshtein import distance as levenshtein_distance

REQUIRED_COLUMNS = {"plate_number", "name", "id", "type"}


class DatabaseError(ValueError):
    """Raised when a vehicle CSV exists but is unsafe or malformed."""


def normalise_plate(value):
    """Return the canonical comparison form used by OCR and CSV records."""
    return "".join(str(value or "").split()).upper()


def load_database(path):
    """
    Read the registered vehicle CSV into a dictionary.

    Expected CSV columns:  plate_number, name, id, type

    Returns
    -------
    dict  { 'PLATENUMBER': {'name': ..., 'id': ..., 'type': ...}, ... }
    """
    database_path = Path(path)
    if not database_path.is_file():
        print(f"Database file '{database_path}' not found.")
        return {}

    db = {}
    with database_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - headers
        if missing:
            raise DatabaseError(
                f"Database is missing required columns: {', '.join(sorted(missing))}"
            )

        for line_number, row in enumerate(reader, start=2):
            key = normalise_plate(row["plate_number"])
            if not key:
                raise DatabaseError(f"Empty plate number on CSV line {line_number}")
            if key in db:
                raise DatabaseError(f"Duplicate plate number: {key}")
            db[key] = {
                "name": row["name"].strip(),
                "id": row["id"].strip(),
                "type": row["type"].strip(),
            }

    print(f"Loaded {len(db)} registered vehicles from '{database_path}'")
    return db


def _match_result(plate, info, distance):
    result = info.copy()
    result["matched_plate"] = plate
    result["distance"] = distance
    return result


def find_match(ocr_text, database, policy="exact", tolerance=1):
    """
    Find the closest database entry for a given OCR result.

    Parameters
    ----------
    ocr_text  : str   – plate text read by OCR
    database  : dict  – loaded by load_database()
    policy    : str   – "exact" (default) or explicitly enabled "fuzzy"
    tolerance : int   – max edit distance when fuzzy matching is enabled

    Returns
    -------
    dict   { name, id, type, matched_plate, distance }  if match found
    None                                                if no match found
    """
    if not ocr_text or not database:
        return None

    query = normalise_plate(ocr_text)
    if not query:
        return None

    if policy == "exact":
        info = database.get(query)
        return _match_result(query, info, 0) if info else None
    if policy != "fuzzy":
        raise ValueError("policy must be 'exact' or 'fuzzy'")
    if tolerance < 0:
        raise ValueError("tolerance must not be negative")

    candidates = [
        (levenshtein_distance(query, plate), plate, info)
        for plate, info in database.items()
    ]
    best_distance = min(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_distance]
    if best_distance <= tolerance and len(best) == 1:
        distance, plate, info = best[0]
        return _match_result(plate, info, distance)

    return None
