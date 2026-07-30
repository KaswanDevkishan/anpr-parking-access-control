import pytest

from matcher import DatabaseError, find_match, load_database, normalise_plate

DATABASE = {
    "ABC123": {"name": "Example", "id": "DEMO-1", "type": "student"},
    "ABC125": {"name": "Sample", "id": "DEMO-2", "type": "staff"},
}


def test_exact_matching_is_default():
    result = find_match("ABC123", DATABASE)

    assert result["matched_plate"] == "ABC123"
    assert result["distance"] == 0


def test_normalisation_removes_whitespace_and_uppercases():
    assert normalise_plate(" ab c\t123\n") == "ABC123"
    assert find_match(" ab c 123 ", DATABASE)["matched_plate"] == "ABC123"


def test_empty_ocr_input_does_not_match():
    assert find_match("", DATABASE) is None
    assert find_match("  ", DATABASE) is None
    assert find_match(None, DATABASE) is None


def test_missing_database_returns_empty_mapping(tmp_path):
    assert load_database(tmp_path / "missing.csv") == {}


def test_invalid_csv_headers_are_rejected(tmp_path):
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text("plate_number,name\nABC123,Example\n", encoding="utf-8")

    with pytest.raises(DatabaseError, match="missing required columns"):
        load_database(csv_path)


def test_duplicate_normalised_plates_are_rejected(tmp_path):
    csv_path = tmp_path / "duplicates.csv"
    csv_path.write_text(
        "plate_number,name,id,type\n"
        "ABC 123,Example,DEMO-1,student\n"
        "abc123,Sample,DEMO-2,staff\n",
        encoding="utf-8",
    )

    with pytest.raises(DatabaseError, match="Duplicate plate number"):
        load_database(csv_path)


def test_ambiguous_fuzzy_tie_is_not_authorized():
    assert find_match("ABC124", DATABASE, policy="fuzzy", tolerance=1) is None


def test_fuzzy_match_outside_tolerance_is_not_authorized():
    assert find_match("XYZ999", DATABASE, policy="fuzzy", tolerance=1) is None
