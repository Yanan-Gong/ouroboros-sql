"""Safe read-only SQL execution.

The safety guarantee lives here, in code, not in agent prompts:
- the connection is opened with SQLite's read-only URI mode, and
- `assert_select_only` requires the statement to parse (via sqlglot) as a
  single SELECT/CTE — DDL, DML, PRAGMA, ATTACH and multi-statement input are
  all rejected before touching the database.
"""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import sqlglot
from sqlglot import expressions as exp

from .introspect import connect_readonly


class UnsafeSQLError(ValueError):
    """Raised when a statement is anything other than a single SELECT."""


class QueryTimeoutError(TimeoutError):
    pass


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool
    elapsed_seconds: float

    def preview(self, max_rows: int = 20) -> str:
        head = [self.columns, *[tuple(str(v) for v in r) for r in self.rows[:max_rows]]]
        lines = [" | ".join(map(str, row)) for row in head]
        if len(self.rows) > max_rows:
            lines.append(f"... ({len(self.rows)} rows total)")
        if self.truncated:
            lines.append("[result truncated at row limit]")
        return "\n".join(lines)


def assert_select_only(sql: str) -> None:
    try:
        statements = sqlglot.parse(sql, dialect="sqlite")
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQLError(f"SQL failed to parse: {e}") from e
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise UnsafeSQLError("Exactly one SQL statement is allowed.")
    root = statements[0]
    if not isinstance(root, (exp.Select, exp.Union)):
        raise UnsafeSQLError(f"Only SELECT queries are allowed, got: {type(root).__name__}")
    banned = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter, exp.Command)
    for node in root.walk():
        if isinstance(node, banned):
            raise UnsafeSQLError(f"Disallowed clause inside query: {type(node).__name__}")


def execute_sql(
    db_path: Path,
    sql: str,
    *,
    timeout_seconds: float = 15.0,
    row_limit: int = 500,
) -> QueryResult:
    """Execute a single SELECT against a read-only connection with a hard timeout."""
    assert_select_only(sql)
    conn = connect_readonly(db_path)
    deadline = time.monotonic() + timeout_seconds

    def _abort_if_past_deadline() -> int:
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(_abort_if_past_deadline, 10_000)
    start = time.monotonic()
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(row_limit + 1)
        columns = [d[0] for d in cursor.description] if cursor.description else []
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e).lower():
            raise QueryTimeoutError(f"Query exceeded {timeout_seconds}s") from e
        raise
    finally:
        conn.close()
    elapsed = time.monotonic() - start
    truncated = len(rows) > row_limit
    return QueryResult(
        columns=columns, rows=rows[:row_limit], truncated=truncated, elapsed_seconds=elapsed
    )


def normalize_result(result: QueryResult, *, order_matters: bool = False) -> tuple:
    """Canonical form for execution-accuracy comparison: a (multi)set of rows
    with floats rounded, column order preserved, row order ignored unless the
    question demands ordering."""

    def norm_value(v: object) -> object:
        if isinstance(v, float):
            return round(v, 6)
        return v

    rows = [tuple(norm_value(v) for v in row) for row in result.rows]
    if not order_matters:
        rows = sorted(rows, key=repr)
    return tuple(rows)


def results_match(pred: QueryResult, gold: QueryResult, *, order_matters: bool = False) -> bool:
    return normalize_result(pred, order_matters=order_matters) == normalize_result(
        gold, order_matters=order_matters
    )
