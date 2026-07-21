"""Function tools for database exploration and (safe) execution.

All tools operate on the database bound into `QueryContext` — agents can never
name an arbitrary file path. Read-only safety is enforced inside `execute_sql`
by the db layer (read-only connection + sqlglot SELECT-only check), not by
prompts.
"""

from dataclasses import dataclass, field
from pathlib import Path

from agents import RunContextWrapper, function_tool

from ..config import settings
from ..db import executor, introspect


@dataclass
class QueryContext:
    """Per-run context shared by all tools."""

    db_id: str
    db_path: Path
    sql_timeout_seconds: float = settings.sql_timeout_seconds
    sql_row_limit: int = settings.sql_row_limit
    # Every execute_sql attempt is recorded here so the runner can identify
    # the final executed query without parsing model output.
    executed_sql: list[dict] = field(default_factory=list)


@function_tool
def list_tables(ctx: RunContextWrapper[QueryContext]) -> str:
    """List all tables in the database, with columns and foreign keys."""
    return introspect.schema_summary(ctx.context.db_path)


@function_tool
def describe_table(ctx: RunContextWrapper[QueryContext], table: str) -> str:
    """Describe one table: columns, types, primary/foreign keys, row count.

    Args:
        table: Exact table name as returned by list_tables.
    """
    t = introspect.describe_table(ctx.context.db_path, table)
    lines = [f"Table {t.name} ({t.row_count} rows)"]
    for c in t.columns:
        flags = "".join([" PK" if c.primary_key else "", " NOT NULL" if c.not_null else ""])
        lines.append(f"  {c.name}: {c.type}{flags}")
    for fk in t.foreign_keys:
        lines.append(f"  FK {t.name}.{fk.column} -> {fk.ref_table}.{fk.ref_column}")
    return "\n".join(lines)


@function_tool
def sample_rows(ctx: RunContextWrapper[QueryContext], table: str, limit: int = 5) -> str:
    """Show a few sample rows from a table to inspect real value formats.

    Args:
        table: Exact table name as returned by list_tables.
        limit: Number of rows to sample (1-20).
    """
    limit = max(1, min(int(limit), 20))
    if table not in introspect.list_tables(ctx.context.db_path):
        return f"Error: no such table {table!r}"
    result = executor.execute_sql(
        ctx.context.db_path,
        f'SELECT * FROM "{table}" LIMIT {limit}',
        timeout_seconds=ctx.context.sql_timeout_seconds,
        row_limit=limit,
    )
    return result.preview(max_rows=limit)


@function_tool
def validate_sql(ctx: RunContextWrapper[QueryContext], sql: str) -> str:
    """Check that a SQL statement parses as a single read-only SELECT, without running it.

    Args:
        sql: The SQL statement to validate.
    """
    try:
        executor.assert_select_only(sql)
    except executor.UnsafeSQLError as e:
        return f"INVALID: {e}"
    return "OK: parses as a single read-only SELECT."


@function_tool
def execute_sql(ctx: RunContextWrapper[QueryContext], sql: str) -> str:
    """Execute a single read-only SELECT and return a preview of the results.

    Args:
        sql: The SELECT statement to run.
    """
    record: dict = {"sql": sql, "ok": False, "error": None}
    ctx.context.executed_sql.append(record)
    try:
        result = executor.execute_sql(
            ctx.context.db_path,
            sql,
            timeout_seconds=ctx.context.sql_timeout_seconds,
            row_limit=ctx.context.sql_row_limit,
        )
    except executor.UnsafeSQLError as e:
        record["error"] = f"unsafe: {e}"
        return f"REJECTED (read-only violation): {e}"
    except executor.QueryTimeoutError as e:
        record["error"] = f"timeout: {e}"
        return f"TIMEOUT: {e}. Simplify the query."
    except Exception as e:  # sqlite errors: missing column, syntax accepted by sqlglot, etc.
        record["error"] = str(e)
        return f"EXECUTION ERROR: {e}"
    record["ok"] = True
    record["row_count"] = len(result.rows)
    header = f"{len(result.rows)} rows in {result.elapsed_seconds:.2f}s"
    return f"{header}\n{result.preview()}"


EXPLORATION_TOOLS = [list_tables, describe_table, sample_rows]
EXECUTION_TOOLS = [validate_sql, execute_sql]
