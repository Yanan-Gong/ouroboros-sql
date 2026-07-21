from pathlib import Path

import pytest

from ouroboros_sql.db.catalog import Catalog, DatabaseNotFoundError
from ouroboros_sql.db.introspect import describe_table, list_tables, schema_summary


def test_catalog_flat_and_nested(tmp_path: Path, tiny_db: Path):
    # flat layout
    flat_root = tiny_db.parent
    cat = Catalog(flat_root)
    assert "schools" in cat.db_ids()
    assert cat.path_for("schools") == tiny_db

    # nested BIRD-style layout
    nested_root = tmp_path / "nested"
    (nested_root / "mydb").mkdir(parents=True)
    (nested_root / "mydb" / "mydb.sqlite").write_bytes(tiny_db.read_bytes())
    cat2 = Catalog(nested_root)
    assert cat2.db_ids() == ["mydb"]

    with pytest.raises(DatabaseNotFoundError):
        cat2.path_for("nope")


def test_list_and_describe(tiny_db: Path):
    assert list_tables(tiny_db) == ["districts", "schools"]
    t = describe_table(tiny_db, "schools")
    assert [c.name for c in t.columns] == [
        "school_id",
        "district_id",
        "name",
        "enrollment",
        "free_meal_rate",
    ]
    assert t.columns[0].primary_key
    assert t.row_count == 3
    assert t.foreign_keys[0].ref_table == "districts"

    with pytest.raises(ValueError):
        describe_table(tiny_db, "sqlite_master")


def test_schema_summary(tiny_db: Path):
    text = schema_summary(tiny_db)
    assert "schools(school_id*" in text
    assert "schools.district_id -> districts.district_id" in text
