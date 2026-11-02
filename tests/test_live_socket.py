"""Live HTTP socket integration test for vexel-eval-harness.

Proves real end-to-end execution over live 127.0.0.1 HTTP sockets:
- Streaming SSE token chunks
- Microsecond TTFT calculation
- Pydantic v2 validation over network payload
- Report generation
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from vexel_eval.schemas import CatalystSignalSchema
from vexel_eval.runner import BenchmarkRunner
from vexel_eval.reporter import BenchmarkReporter
from mock_server import run_mock_server


def test_live_socket_benchmark(tmp_path):
    """Run real streaming HTTP benchmark against local mock LLM server."""
    port = 8899
    server = run_mock_server(port)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.1)

    try:
        dataset_path = Path(__file__).parent.parent / "benchmarks" / "sample_catalysts_50.jsonl"
        samples = BenchmarkRunner.load_dataset(dataset_path)

        runner = BenchmarkRunner(
            endpoint=f"http://127.0.0.1:{port}/v1",
            model_name="vexel-qwen-2.5-1.5b-q4_k_m",
            concurrency=2,
            simulate=False,  # REAL LIVE HTTP REQUESTS
        )

        async def _exec():
            return await runner.run_benchmark(
                samples=samples[:5],
                schema_cls=CatalystSignalSchema,
                profile="vexel-fine-tuned",
                custom_model_name="vexel-qwen-2.5-1.5b-q4_k_m",
            )

        # Run async benchmark
        metrics = asyncio.run(_exec())

        assert metrics.total_samples == 5
        assert metrics.valid_json_count == 5
        assert metrics.schema_compliant_count == 5
        assert metrics.schema_compliance_rate_pct == 100.0
        assert metrics.raw_json_rate_pct == 100.0
        # Verify real socket TTFT and latency
        assert metrics.ttft_p50_ms > 0
        assert metrics.latency_p50_ms > metrics.ttft_p50_ms
        assert metrics.overall_field_accuracy_pct == 100.0

        # Verify Markdown report writing
        report_file = tmp_path / "LIVE_BENCHMARK_REPORT.md"
        reporter = BenchmarkReporter()
        generated = reporter.generate_markdown_report(metrics, output_path=report_file)
        assert generated.exists()
        content = generated.read_text(encoding="utf-8")
        assert "vexel-qwen-2.5-1.5b-q4_k_m" in content
        assert "TTFT" in content

    finally:
        server.shutdown()
        server.server_close()
