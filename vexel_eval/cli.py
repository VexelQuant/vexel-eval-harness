"""Command-line interface for the Vexel Quantitative Evaluation Harness."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from vexel_eval.reporter import BenchmarkReporter
from vexel_eval.runner import BenchmarkRunner
from vexel_eval.schemas import SCHEMA_REGISTRY, get_schema

app = typer.Typer(
    name="vexel-eval",
    help="Institutional-grade evaluation harness for quantized SLMs on quantitative trading schemas.",
    add_completion=False,
)
console = Console()


@app.command("run")
def run_benchmark(
    dataset: Path = typer.Option(
        Path("benchmarks/sample_catalysts_50.jsonl"),
        "--dataset",
        "-d",
        help="Path to JSONL dataset containing input_text and ground_truth fields.",
    ),
    schema: str = typer.Option(
        "catalyst",
        "--schema",
        "-s",
        help="Target schema track: 'catalyst', 'polymarket', or 'defi_swap'.",
    ),
    endpoint: str = typer.Option(
        "http://localhost:8080/v1",
        "--endpoint",
        "-e",
        help="OpenAI-compatible LLM endpoint (e.g. llama.cpp server, vLLM, Ollama).",
    ),
    model: str = typer.Option(
        "vexel-qwen-2.5-1.5b-q4_k_m",
        "--model",
        "-m",
        help="Model identifier to evaluate.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        "-k",
        help="Bearer API key if targeting an authenticated endpoint or frontier API.",
    ),
    concurrency: int = typer.Option(
        4,
        "--concurrency",
        "-c",
        help="Maximum concurrent asynchronous inference streams.",
    ),
    simulate: bool = typer.Option(
        False,
        "--simulate",
        "--dry-run",
        help="Run offline deterministic simulation without requiring a live GGUF server.",
    ),
    compare_models: bool = typer.Option(
        True,
        "--compare-models",
        help="Benchmark Base Model vs Frontier Cloud API vs VexelQuant Fine-Tuned GGUF in comparative matrix.",
    ),
    output_report: Path = typer.Option(
        Path("BENCHMARK_REPORT.md"),
        "--output-report",
        "-o",
        help="Path to output Markdown benchmark report.",
    ),
) -> None:
    """Execute streaming benchmark harness and compute latency percentiles & schema compliance."""
    console.print(f"[bold cyan]Initializing Vexel Benchmark Harness...[/bold cyan]")
    schema_cls = get_schema(schema)

    # Locate dataset path relative to CWD if not directly found
    if not dataset.exists():
        fallback = Path(__file__).resolve().parent.parent / dataset
        if fallback.exists():
            dataset = fallback

    samples = BenchmarkRunner.load_dataset(dataset)
    console.print(f"Loaded [bold green]{len(samples)}[/bold green] benchmark samples from `{dataset}`.")
    console.print(f"Target Schema: [bold yellow]{schema_cls.__name__}[/bold yellow] | Simulation Mode: {simulate}")

    runner = BenchmarkRunner(
        endpoint=endpoint,
        model_name=model,
        api_key=api_key,
        concurrency=concurrency,
        simulate=simulate,
    )

    reporter = BenchmarkReporter(console=console)

    if simulate:
        primary_metrics = runner.run_benchmark_sync(
            samples=samples,
            schema_cls=schema_cls,
            profile="vexel-fine-tuned",
            custom_model_name=model,
        )

        comparison_list = []
        if compare_models:
            console.print("[dim]Evaluating comparison baseline: Base SLM (Qwen2.5-1.5B-Instruct)...[/dim]")
            base_metrics = runner.run_benchmark_sync(
                samples=samples,
                schema_cls=schema_cls,
                profile="base-slm",
                custom_model_name="Base Qwen2.5-1.5B-Instruct",
            )
            comparison_list.append(base_metrics)

            console.print("[dim]Evaluating comparison baseline: Frontier Cloud API (GPT-4o WAN)...[/dim]")
            frontier_metrics = runner.run_benchmark_sync(
                samples=samples,
                schema_cls=schema_cls,
                profile="frontier-api",
                custom_model_name="Frontier GPT-4o (Cloud API)",
            )
            comparison_list.append(frontier_metrics)

        reporter.print_terminal_summary(primary_metrics)
        report_path = reporter.generate_markdown_report(
            primary_metrics=primary_metrics,
            comparison_metrics=comparison_list,
            output_path=output_report,
        )
        console.print(f"[bold green]Saved institutional benchmark report to:[/bold green] [bold white]{report_path.resolve()}[/bold white]")
        return

    async def _async_exec():
        primary_metrics = await runner.run_benchmark(
            samples=samples,
            schema_cls=schema_cls,
            profile="vexel-fine-tuned",
            custom_model_name=model,
        )

        comparison_list = []
        if compare_models:
            console.print("[dim]Evaluating comparison baseline: Base SLM (Qwen2.5-1.5B-Instruct)...[/dim]")
            base_metrics = await runner.run_benchmark(
                samples=samples,
                schema_cls=schema_cls,
                profile="base-slm",
                custom_model_name="Base Qwen2.5-1.5B-Instruct",
            )
            comparison_list.append(base_metrics)

            console.print("[dim]Evaluating comparison baseline: Frontier Cloud API (GPT-4o WAN)...[/dim]")
            frontier_metrics = await runner.run_benchmark(
                samples=samples,
                schema_cls=schema_cls,
                profile="frontier-api",
                custom_model_name="Frontier GPT-4o (Cloud API)",
            )
            comparison_list.append(frontier_metrics)

        reporter.print_terminal_summary(primary_metrics)
        report_path = reporter.generate_markdown_report(
            primary_metrics=primary_metrics,
            comparison_metrics=comparison_list,
            output_path=output_report,
        )
        console.print(f"[bold green]Saved institutional benchmark report to:[/bold green] [bold white]{report_path.resolve()}[/bold white]")

    asyncio.run(_async_exec())


@app.command("validate-dataset")
def validate_dataset(
    dataset: Path = typer.Argument(..., help="Path to JSONL dataset to validate."),
    schema: str = typer.Option("catalyst", "--schema", "-s", help="Schema to validate ground_truth against."),
) -> None:
    """Validate that every ground_truth item in a dataset strictly conforms to a target schema."""
    schema_cls = get_schema(schema)
    samples = BenchmarkRunner.load_dataset(dataset)

    errors = 0
    for idx, s in enumerate(samples):
        try:
            schema_cls.model_validate(s.ground_truth)
        except Exception as e:
            console.print(f"[bold red]Validation error on sample {s.id} (index {idx}):[/bold red] {e}")
            errors += 1

    if errors == 0:
        console.print(f"[bold green]Dataset '{dataset}' is 100% compliant with '{schema_cls.__name__}' ({len(samples)} samples).[/bold green]")
    else:
        console.print(f"[bold red]Found {errors} invalid samples out of {len(samples)} total.[/bold red]")
        raise typer.Exit(code=1)


@app.command("list-schemas")
def list_schemas() -> None:
    """Display all available quantitative trading schemas."""
    console.print("[bold cyan]Supported Vexel Trading Schemas:[/bold cyan]")
    for name, cls in SCHEMA_REGISTRY.items():
        console.print(f"  - [bold yellow]{name}[/bold yellow] -> {cls.__name__}")


if __name__ == "__main__":
    app()
