"""Run live HTTP server and execute vexel-eval CLI over real HTTP sockets."""

import subprocess
import sys
import time
import threading
from pathlib import Path

# Add tests directory to sys.path to import mock_server
harness_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(harness_dir / "tests"))
from mock_server import run_mock_server

def main():
    port = 8765
    server = run_mock_server(port)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[1/3] Started live mock LLM HTTP server on http://127.0.0.1:{port}", flush=True)
    time.sleep(0.5)

    try:
        dataset_file = harness_dir / "benchmarks" / "sample_catalysts_50.jsonl"
        report_file = harness_dir / "BENCHMARK_REPORT.md"

        cmd = [
            sys.executable,
            "-m",
            "vexel_eval.cli",
            "run",
            "--dataset", str(dataset_file),
            "--schema", "catalyst",
            "--endpoint", f"http://127.0.0.1:{port}/v1",
            "--model", "vexel-qwen-2.5-1.5b-q4_k_m",
            "--concurrency", "4",
            "--output-report", str(report_file),
        ]

        print(f"[2/3] Executing CLI live benchmark: {' '.join(cmd)}", flush=True)
        res = subprocess.run(cmd, cwd=str(harness_dir), capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print("STDERR:", res.stderr)

        if res.returncode != 0:
            print(f"CLI command failed with code {res.returncode}")
            sys.exit(res.returncode)

        print(f"[3/3] Live benchmark completed! Verifying {report_file}...", flush=True)
        if not report_file.exists():
            raise FileNotFoundError(f"Report file {report_file} was not generated")

        content = report_file.read_text(encoding="utf-8")
        assert "vexel-qwen-2.5-1.5b-q4_k_m" in content
        assert "TTFT" in content
        assert "Executive Summary" in content
        print(f"SUCCESS: BENCHMARK_REPORT.md verified with {len(content)} bytes.")

    finally:
        server.shutdown()
        server.server_close()
        print("Stopped live mock LLM HTTP server.")

if __name__ == "__main__":
    main()
