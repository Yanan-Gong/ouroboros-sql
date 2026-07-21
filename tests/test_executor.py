from pathlib import Path

import pytest

from ouroboros_sql.db.executor import (
    QueryTimeoutError,
    UnsafeSQLError,
    assert_select_only,
    execute_sql,
    normalize_result,
    results_match,
)


class TestSelectOnlyEnforcement:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM schools",
            "SELECT name FROM schools WHERE enrollment > 500 ORDER BY name",
            "WITH big AS (SELECT * FROM schools WHERE enrollment > 500) SELECT name FROM big",
            "SELECT name FROM schools UNION SELECT name FROM districts",
        ],
    )
    def test_allows_selects(self, sql: str):
        assert_select_only(sql)

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE schools",
            "DELETE FROM schools",
            "UPDATE schools SET enrollment = 0",
            "INSERT INTO schools VALUES (1, 1, 'x', 1, 0.1)",
            "CREATE TABLE t (x INT)",
            "ALTER TABLE schools ADD COLUMN hacked INT",
            "PRAGMA writable_schema = 1",
            "ATTACH DATABASE '/tmp/evil.db' AS evil",
            "SELECT 1; DROP TABLE schools",
            "not sql at all (",
        ],
    )
    def test_rejects_everything_else(self, sql: str):
        with pytest.raises(UnsafeSQLError):
            assert_select_only(sql)

    def test_writes_blocked_even_bypassing_parser(self, tiny_db: Path):
        # Belt and suspenders: even if the AST check were fooled, the
        # connection itself is read-only.
        import sqlite3

        from ouroboros_sql.db.introspect import connect_readonly

        conn = connect_readonly(tiny_db)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM schools")
        conn.close()


class TestExecution:
    def test_basic_query(self, tiny_db: Path):
        result = execute_sql(tiny_db, "SELECT name FROM schools ORDER BY school_id")
        assert result.columns == ["name"]
        assert [r[0] for r in result.rows] == ["Bay High", "Shore Elementary", "Valley Middle"]
        assert not result.truncated

    def test_row_limit_truncates(self, tiny_db: Path):
        result = execute_sql(tiny_db, "SELECT * FROM schools", row_limit=2)
        assert len(result.rows) == 2
        assert result.truncated

    def test_timeout(self, tiny_db: Path):
        # Recursive CTE that never finishes.
        endless = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM r) SELECT * FROM r"
        with pytest.raises(QueryTimeoutError):
            execute_sql(tiny_db, endless, timeout_seconds=0.2, row_limit=10**9)


class TestNormalization:
    def test_row_order_ignored_by_default(self, tiny_db: Path):
        a = execute_sql(tiny_db, "SELECT name FROM schools ORDER BY name ASC")
        b = execute_sql(tiny_db, "SELECT name FROM schools ORDER BY name DESC")
        assert results_match(a, b)
        assert not results_match(a, b, order_matters=True)

    def test_float_rounding(self, tiny_db: Path):
        a = execute_sql(tiny_db, "SELECT 0.1 + 0.2")
        b = execute_sql(tiny_db, "SELECT 0.3")
        assert results_match(a, b)

    def test_different_results_do_not_match(self, tiny_db: Path):
        a = execute_sql(tiny_db, "SELECT name FROM schools")
        b = execute_sql(tiny_db, "SELECT name FROM districts")
        assert normalize_result(a) != normalize_result(b)
