#!/usr/bin/env python3
"""Seed the strategy memory from the baseline-val-v0 failure analysis.

Hand-written entries, each grounded in the failure taxonomy and exemplars from
`docs/results/baseline-val-v0/` (see analysis.md). This is the M3 human
baseline for memory content; from M4 on, the optimizer evolves it.
"""

from ouroboros_sql.agents.memory import MemoryEntry, StrategyMemory

PROV = ["baseline-val-v0"]

ENTRIES = [
    # --- SQLWriter: wrong_result was the top class (78 records) ------------
    MemoryEntry(
        id="writer-constraint-checklist",
        kind="heuristic",
        scope="sql_writer",
        text=(
            "Before emitting SQL, enumerate every explicit constraint in the "
            "question (years, names, nationalities, ids, categories) and check "
            "each one appears as a WHERE or JOIN predicate. A missing filter "
            "aggregates over the wrong population."
        ),
        provenance=[*PROV, "wrong_result"],
    ),
    MemoryEntry(
        id="writer-ratio-same-population",
        kind="heuristic",
        scope="sql_writer",
        text=(
            "For percentages and ratios, build numerator and denominator from "
            "the same filtered set and CAST to REAL: "
            "SUM(CASE WHEN cond THEN 1.0 ELSE 0 END) / COUNT(*) over identical "
            "rows. Integer division and mismatched denominators are the top "
            "causes of wrong results."
        ),
        provenance=[*PROV, "wrong_result"],
    ),
    MemoryEntry(
        id="writer-only-requested-columns",
        kind="heuristic",
        scope="sql_writer",
        text=(
            "SELECT exactly the columns the question asks for — no extras, in "
            "the asked order. Result comparison is strict about shape."
        ),
        provenance=[*PROV, "wrong_result"],
    ),
    MemoryEntry(
        id="writer-no-clarification-stall",
        kind="pitfall",
        scope="sql_writer",
        text=(
            "Never stall asking which interpretation the user meant. Adopt the "
            "standard reading ('X% higher than average' means > AVG*(1+X/100); "
            "'from YEAR to YEAR' is inclusive BETWEEN) and proceed."
        ),
        provenance=[*PROV, "no_sql_executed"],
    ),
    # --- SchemaLinker: wrong_tables (26 records) ----------------------------
    MemoryEntry(
        id="linker-exact-column-spelling",
        kind="heuristic",
        scope="schema_linker",
        text=(
            "Pass the SQLWriter the exact column spellings from describe_table "
            "(including casing and spaces). If a name looks abbreviated or "
            "ambiguous, confirm against the sample_rows header before handing "
            "off; never let the writer guess column names."
        ),
        provenance=[*PROV, "wrong_tables"],
    ),
    MemoryEntry(
        id="linker-sample-filter-columns",
        kind="heuristic",
        scope="schema_linker",
        text=(
            "Call sample_rows on every column that will be filtered on, and "
            "report the observed value formats (case, codes, units) in the "
            "handoff so filters match stored values exactly."
        ),
        provenance=[*PROV, "wrong_result"],
    ),
    # --- Validator: no_sql_executed (19 records) ----------------------------
    MemoryEntry(
        id="validator-always-execute",
        kind="heuristic",
        scope="validator",
        text=(
            "Never finish without at least one executed query. If the result "
            "is empty or implausible, retry once with corrected filter values "
            "before accepting it."
        ),
        provenance=[*PROV, "no_sql_executed"],
    ),
    MemoryEntry(
        id="validator-error-detail-on-bounce",
        kind="pitfall",
        scope="validator",
        text=(
            "When bouncing back to the SQLWriter, always include the exact "
            "error message or a one-sentence diagnosis — a bare bounce wastes "
            "a retry."
        ),
        provenance=[*PROV, "sql_never_recovered"],
    ),
    # --- Summarizer ----------------------------------------------------------
    MemoryEntry(
        id="summarizer-direct-values",
        kind="heuristic",
        scope="summarizer",
        text=(
            "Lead with the direct answer using only values present in the "
            "executed result rows; never round or reformat numbers unless "
            "asked."
        ),
        provenance=PROV,
    ),
]


def main() -> None:
    memory = StrategyMemory.load()
    for entry in ENTRIES:
        memory.upsert(entry)
    memory.save()
    print(f"Seeded {len(ENTRIES)} entries -> {memory.stats()}")


if __name__ == "__main__":
    main()
