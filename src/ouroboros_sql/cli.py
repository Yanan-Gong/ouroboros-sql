"""Command-line interface: `ouroboros <command>`."""

import asyncio

import typer
from rich.console import Console
from rich.panel import Panel

from .config import settings

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def query(
    question: str = typer.Argument(None, help="Analytics question (omit with --interactive)"),
    db: str = typer.Option(..., "--db", help="Database id, e.g. california_schools"),
    interactive: bool = typer.Option(False, help="Multi-turn session with follow-ups"),
    verbose: bool = typer.Option(False, "-v", help="Show the trajectory (tools, handoffs)"),
) -> None:
    """Answer an analytics question over one of the downloaded databases."""
    from agents.memory import SQLiteSession

    from .agents.topology import build_pipeline
    from .bootstrap import configure_openai
    from .runner import run_one

    configure_openai()
    pipeline = build_pipeline()
    session = SQLiteSession(f"cli-{db}") if interactive else None

    async def ask(q: str) -> None:
        record = await run_one(q, db, pipeline, session=session)
        if verbose:
            for e in record.events:
                if e.kind == "tool_call":
                    console.print(f"[dim]  {e.agent} → {e.payload['tool']}[/dim]")
                elif e.kind == "handoff":
                    console.print(
                        f"[dim]  handoff {e.payload['source']} → {e.payload['target']}[/dim]"
                    )
        style = "green" if record.status == "ok" else "yellow"
        console.print(Panel(record.final_output, border_style=style))
        console.print(
            f"[dim]{record.status} · {record.requests} model calls · "
            f"{record.input_tokens}+{record.output_tokens} tokens · "
            f"{record.wall_seconds:.1f}s[/dim]"
        )

    if interactive:
        console.print(f"[bold]Interactive session on {db!r}[/bold] (empty line to exit)")
        while True:
            q = console.input("[bold cyan]? [/bold cyan]").strip()
            if not q:
                break
            asyncio.run(ask(q))
    else:
        if not question:
            raise typer.BadParameter("Provide a question or use --interactive.")
        asyncio.run(ask(question))


@app.command("list-dbs")
def list_dbs() -> None:
    """List downloaded databases."""
    from .db.catalog import Catalog

    ids = Catalog(settings.databases_dir).db_ids()
    if not ids:
        console.print("No databases found. Run: [bold]ouroboros download-data[/bold]")
        return
    for db_id in ids:
        console.print(db_id)


@app.command()
def eval(
    split: str = typer.Option("val", help="Golden split: train | val | holdout"),
    repeats: int = typer.Option(8, help="Repeat runs per example (reliability decomposition)"),
    concurrency: int = typer.Option(8, help="Concurrent pipeline runs"),
    limit: int = typer.Option(None, help="Only the first N examples (smoke tests)"),
    judge: bool = typer.Option(False, help="Also score trajectories with the LLM judge"),
    run_id: str = typer.Option(None, help="Resume/name a run directory"),
) -> None:
    """Run the golden set through the pipeline and compute trajectory metrics."""
    from .bootstrap import configure_openai
    from .eval.harness import run_eval
    from .eval.schema import load_split, read_jsonl
    from .eval.tables import to_markdown
    from .eval.taxonomy import taxonomy_counts

    configure_openai()
    metrics, run_dir = asyncio.run(
        run_eval(
            split,
            repeats=repeats,
            concurrency=concurrency,
            limit=limit,
            with_judge=judge,
            run_id=run_id,
        )
    )
    markdown = to_markdown(metrics, run_dir.name)
    (run_dir / "results.md").write_text(markdown + "\n")
    console.print(markdown)

    examples = load_split(split)  # type: ignore[arg-type]
    counts = taxonomy_counts(read_jsonl(run_dir / "records.jsonl"), examples)
    if counts:
        console.print("\n[bold]Failure taxonomy[/bold]")
        for label, count in counts.items():
            console.print(f"  {label}: {count}")
    console.print(f"\n[dim]artifacts: {run_dir}[/dim]")


@app.command("download-data")
def download_data() -> None:
    """Download the BIRD mini-dev SQLite databases (checksummed)."""
    from .data_setup import main as download_main

    download_main()


if __name__ == "__main__":
    app()
