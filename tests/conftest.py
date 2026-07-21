import sqlite3
from pathlib import Path

import pytest


@pytest.fixture()
def tiny_db(tmp_path: Path) -> Path:
    """A small schools database mirroring the shape of BIRD-style benchmarks."""
    db_path = tmp_path / "schools.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE districts (
            district_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            county TEXT
        );
        CREATE TABLE schools (
            school_id INTEGER PRIMARY KEY,
            district_id INTEGER REFERENCES districts(district_id),
            name TEXT NOT NULL,
            enrollment INTEGER,
            free_meal_rate REAL
        );
        INSERT INTO districts VALUES (1, 'Alameda USD', 'Alameda'), (2, 'Fresno USD', 'Fresno');
        INSERT INTO schools VALUES
            (10, 1, 'Bay High', 1200, 0.42),
            (11, 1, 'Shore Elementary', 450, 0.61),
            (12, 2, 'Valley Middle', 800, 0.55);
        """
    )
    conn.commit()
    conn.close()
    return db_path
