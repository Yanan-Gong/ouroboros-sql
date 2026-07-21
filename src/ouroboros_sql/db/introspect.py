"""Schema introspection via SQLite PRAGMAs. Pure stdlib; read-only connections."""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Column:
    name: str
    type: str
    primary_key: bool = False
    not_null: bool = False


@dataclass
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass
class Table:
    name: str
    columns: list[Column] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    row_count: int | None = None


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return conn


def list_tables(db_path: Path) -> list[str]:
    with connect_readonly(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            " ORDER BY name"
        ).fetchall()
    return [r[0] for r in rows]


def describe_table(db_path: Path, table: str, with_row_count: bool = True) -> Table:
    if table not in list_tables(db_path):
        raise ValueError(f"No such table: {table!r}")
    with connect_readonly(db_path) as conn:
        cols = [
            Column(name=r[1], type=r[2] or "", not_null=bool(r[3]), primary_key=bool(r[5]))
            for r in conn.execute(f'PRAGMA table_info("{table}")')
        ]
        fks = [
            ForeignKey(column=r[3], ref_table=r[2], ref_column=r[4] or "")
            for r in conn.execute(f'PRAGMA foreign_key_list("{table}")')
        ]
        count = None
        if with_row_count:
            count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    return Table(name=table, columns=cols, foreign_keys=fks, row_count=count)


def schema_summary(db_path: Path) -> str:
    """Compact one-line-per-table summary used in agent prompts."""
    lines = []
    for name in list_tables(db_path):
        t = describe_table(db_path, name, with_row_count=False)
        cols = ", ".join(c.name + ("*" if c.primary_key else "") for c in t.columns)
        lines.append(f"{name}({cols})")
        for fk in t.foreign_keys:
            lines.append(f"  {name}.{fk.column} -> {fk.ref_table}.{fk.ref_column}")
    return "\n".join(lines)
