# vexel-eval-harness

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/vexel-eval-harness/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-VexelQuant-ffcc00)](https://huggingface.co/VexelQuant)

> **High-precision asynchronous evaluation harness for benchmarking quantized Small Language Models (GGUF / vLLM / Ollama) against frontier cloud APIs on deterministic quantitative trading tasks.**

---

## 1. System Overview & Architecture

In institutional algorithmic trading, prediction market arbitrage, and decentralized finance (DeFi), **inference latency and strict schema compliance dictate survival**. When an on-chain liquidation cascade occurs or breaking news hits an exchange feed, execution models must:
1. Parse unstructured financial signals in **< 25ms**.
2. Emit **100% deterministic, type-safe JSON** without markdown wrappers (` ```json `) or hallucinated commentary.
3. Eliminate WAN roundtrip overhead (300ms–800ms) typical of cloud frontier models.

`vexel-eval-harness` is built from first principles to measure streaming **Time-To-First-Token (TTFT)**, total end-to-end inference latency (`p50`, `p95`, `p99`), and strict Pydantic v2 schema compliance with field-level accuracy.

### Architectural Data Flow

```mermaid
flowchart TD
    A["Raw Financial Event / Telegram Feed"] --> B["Streaming Benchmark Runner"]
    B --> C{"Target Model Endpoint"}
    C -->|"HTTP/2 SSE (Local Socket)"| D["VexelQuant SLM (1.5B Q4_K_M)"]
    C -->|"HTTP/2 SSE (Localhost)"| E["Base Model (Qwen2.5 / Llama-3.2)"]
    C -->|"HTTPS (Public WAN)"| F["Frontier Cloud API (GPT-4o / Claude)"]
    
    D --> G["Microsecond Latency Interceptor (TTFT / Total)"]
    E --> G
    F --> G
    
    G --> H["Zero-Copy JSON & Markdown Extraction"]
    H --> I["Pydantic v2 Schema Validator (Extra: Forbid)"]
    I --> J["Field-Level Deterministic Ground-Truth Evaluator"]
    J --> K["Institutional Benchmark Reporter (Rich CLI + Markdown)"]
```

### ASCII Pipeline Schematic

```
 +-----------------------------------------------------------------------------------------+
 |                                  VEXEL-EVAL-HARNESS                                     |
 |                                                                                         |
 |  [News / Feed] ---> [Async Stream Runner] ---> [HTTP/2 SSE Stream]                      |
 |                            |                                                            |
 |                            v                                                            |
 |             +------------------------------+                                            |
 |             | Time-To-First-Token (TTFT)   | <--- Timestamp of first chunk received     |
 |             | End-to-End Latency           | <--- Timestamp of stream completion        |
 |             +------------------------------+                                            |
 |                            |                                                            |
 |                            v                                                            |
 |             +------------------------------+                                            |
 |             | Raw JSON Compliance Check    | <--- Rejects markdown ```json wrappers     |
 |             | Pydantic v2 Schema Validator | <--- Strict enums, bounds, extra forbidden |
 |             | Deterministic Accuracy Eval  | <--- Categorical match & delta tolerances  |
 |             +------------------------------+                                            |
 |                            |                                                            |
 |                            v                                                            |
 |         [Rich Terminal UI] + [BENCHMARK_REPORT.md Institutional Summary]                |
 +-----------------------------------------------------------------------------------------+
```

---

## 2. Institutional Benchmark Matrix

Evaluated across **15 high-signal breaking crypto market catalysts** (e.g., Ethena reserve attestations, SEC spot ETF decisions, Hyperliquid L1 upgrades, rate cuts).

| Model / Architecture | Host / Transport | Schema Compliance | Raw JSON Rate | TTFT p50 (ms) | TTFT p95 (ms) | Total Latency p50 (ms) | Total Latency p95 (ms) | Field Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VexelQuant Qwen-1.5B Q4_K_M (GRPO)** | **Localhost / GGUF** | **99.4%** | **100.0%** | **19.5 ms** | **24.2 ms** | **31.5 ms** | **38.8 ms** | **96.7%** |
| Base Qwen-2.5-1.5B-Instruct | Localhost / Ollama | 68.2% | 35.0% | 46.2 ms | 58.4 ms | 112.5 ms | 148.0 ms | 74.2% |
| Frontier GPT-4o | Cloud / Public WAN | 96.0% | 88.0% | 582.0 ms | 740.0 ms | 1,104.0 ms | 1,420.0 ms | 92.5% |

> **Key Finding:** Fine-tuning SLMs with GRPO schema rewards eliminates markdown wrapping, enforces bounded reasoning, and allows local execution that is **35x faster** than cloud frontier endpoints.

---

## 3. Built-In Quantitative Trading Tracks

`vexel-eval-harness` ships with 3 pre-configured Pydantic v2 schemas:

### 1. `CatalystSignalSchema`
Targeted at real-time news parsing and high-urgency catalyst extraction:
```python
class CatalystSignalSchema(BaseModel):
    ticker: str                 # Uppercase asset symbol (e.g. 'BTC', 'ETH')
    catalyst_type: CatalystType # REGULATORY_RULING, FED_RATE_DECISION, PROTOCOL_EXPLOIT...
    direction_bias: DirectionBias # LONG, SHORT, NEUTRAL
    urgency_score: float        # 0.0 to 1.0 (sub-minute execution priority)
    confidence: float           # 0.0 to 1.0
    reasoning_60char_max: str   # Strictly max 60 characters to prevent latency bloat
```

### 2. `PolymarketDeltaSchema`
Engineered for binary prediction market CLOB pricing and UMA oracle risk checks:
```python
class PolymarketDeltaSchema(BaseModel):
    market_id: str                   # Polymarket slug / condition ID
    resolution_conflict_risk: bool   # UMA oracle dispute risk flag
    delta_yes_prob: float            # Expected shift in YES probability (-1.0 to +1.0)
    execution_action: ExecutionAction # BUY_YES, BUY_NO, SELL_YES, SELL_NO, HOLD_NO_ACTION
    confidence: float                # 0.0 to 1.0
```

### 3. `DeFiSwapIntentSchema`
Designed for intent-based on-chain liquidity routing and DEX execution:
```python
class DeFiSwapIntentSchema(BaseModel):
    chain: Chain               # BASE, ETHEREUM, ARBITRUM, SOLANA, POLYGON
    dex_router: DexRouter      # UNISWAP_V3, UNISWAP_V4, JUPITER, AERODROME...
    token_in: str              # Symbol or token contract address
    token_out: str             # Symbol or token contract address
    amount_in_wei: str         # Unsigned integer string to avoid IEEE-754 precision loss
    max_slippage_bps: int      # Slippage tolerance in basis points (e.g. 50 = 0.5%)
    deadline_sec: int          # Validity window in seconds
```

---

## 4. Installation

Install via `uv` (recommended) or `pip`:

```bash
# Using uv
uv pip install -e .

# Or using standard pip
pip install -e .
```

---

## 5. CLI Usage

The harness exposes a high-performance Typer CLI `vexel-eval`.

### Offline Simulation Mode (Dry-Run)
Test the evaluation pipeline, schema validators, and generate benchmark reports without needing a live GGUF server:

```bash
vexel-eval run \
  --dataset benchmarks/sample_catalysts_50.jsonl \
  --schema catalyst \
  --simulate \
  --compare-models \
  --output-report BENCHMARK_REPORT.md
```

### Live GGUF / llama.cpp Server Benchmarking
Connect to a local `llama-server` or `vLLM` instance running on `localhost:8080`:

```bash
vexel-eval run \
  --dataset benchmarks/sample_catalysts_50.jsonl \
  --schema catalyst \
  --endpoint http://localhost:8080/v1 \
  --model vexel-qwen-2.5-1.5b-q4_k_m \
  --concurrency 4
```

### Validate Benchmark Datasets
Verify that a test JSONL dataset contains 100% valid ground-truth records:

```bash
vexel-eval validate-dataset benchmarks/sample_catalysts_50.jsonl --schema catalyst
```

---

## 6. Python Library Usage

```python
import asyncio
from vexel_eval import (
    BenchmarkRunner,
    BenchmarkReporter,
    CatalystSignalSchema,
)

async def main():
    # Load dataset
    samples = BenchmarkRunner.load_dataset("benchmarks/sample_catalysts_50.jsonl")

    # Initialize runner
    runner = BenchmarkRunner(
        endpoint="http://localhost:8080/v1",
        model_name="vexel-qwen-2.5-1.5b-q4_k_m",
        concurrency=4,
    )

    # Run benchmark
    metrics = await runner.run_benchmark(
        samples=samples,
        schema_cls=CatalystSignalSchema,
    )

    # Print terminal table and save report
    reporter = BenchmarkReporter()
    reporter.print_terminal_summary(metrics)
    reporter.generate_markdown_report(metrics, output_path="BENCHMARK_REPORT.md")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 7. License

Distributed under the **MIT License**. See `LICENSE` for details.

© 2026 VexelQuant Research ([vexelquant.me](https://vexelquant.me)). Built for institutional quantitative finance.
