"""High-precision asynchronous benchmark runner for quantized SLM evaluation."""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import httpx
import numpy as np
from pydantic import BaseModel, ValidationError

from vexel_eval.schemas import (
    CatalystSignalSchema,
    DeFiSwapIntentSchema,
    PolymarketDeltaSchema,
    get_schema,
    get_system_instruction,
)


@dataclass
class BenchmarkSample:
    """Single benchmark prompt and deterministic ground-truth payload."""
    id: str
    input_text: str
    ground_truth: Dict[str, Any]


@dataclass
class SampleResult:
    """Granular execution telemetry for a single prompt."""
    sample_id: str
    prompt: str
    raw_text: str
    parsed_json: Optional[Dict[str, Any]]
    is_valid_json: bool
    is_raw_json: bool  # True if valid JSON without markdown wrapping
    is_schema_compliant: bool
    validation_error: Optional[str]
    ttft_ms: float
    total_latency_ms: float
    tokens_generated: int
    field_matches: Dict[str, bool]
    ground_truth: Dict[str, Any]


@dataclass
class ModelBenchmarkMetrics:
    """Aggregated statistical evaluation summary for a model."""
    model_name: str
    schema_name: str
    total_samples: int
    valid_json_count: int
    raw_json_count: int
    valid_json_rate_pct: float
    raw_json_rate_pct: float
    schema_compliant_count: int
    schema_compliance_rate_pct: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    field_accuracies_pct: Dict[str, float]
    overall_field_accuracy_pct: float
    avg_tokens_per_sec: float
    results: List[SampleResult] = field(default_factory=list)


def extract_json_candidate(text: str) -> Tuple[Optional[Dict[str, Any]], bool, bool]:
    """Parse text into JSON and determine whether it was pure raw JSON or markdown wrapped.

    Returns:
        (parsed_dict, is_valid_json, is_raw_json)
    """
    clean_text = text.strip()
    if not clean_text:
        return None, False, False

    # Check 1: Direct raw JSON without markdown fences
    if not ("```" in clean_text):
        try:
            parsed = json.loads(clean_text)
            if isinstance(parsed, dict):
                return parsed, True, True
        except Exception:
            pass

    # Check 2: Markdown wrapped JSON ```json ... ```
    md_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(md_pattern, clean_text)
    if match:
        snippet = match.group(1).strip()
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed, True, False
        except Exception:
            pass

    # Check 3: Extract outermost brace { ... }
    brace_start = clean_text.find("{")
    brace_end = clean_text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        snippet = clean_text[brace_start : brace_end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed, True, False
        except Exception:
            pass

    return None, False, False


def evaluate_field_accuracy(
    predicted: Dict[str, Any], ground_truth: Dict[str, Any]
) -> Dict[str, bool]:
    """Compare prediction against ground-truth with strict type-specific deterministic logic."""
    matches: Dict[str, bool] = {}
    for key, expected in ground_truth.items():
        if key not in predicted:
            matches[key] = False
            continue

        val = predicted[key]
        if isinstance(expected, float) or isinstance(val, float):
            # Numerical proximity tolerance: <= 0.05
            try:
                diff = abs(float(val) - float(expected))
                matches[key] = diff <= 0.05
            except (ValueError, TypeError):
                matches[key] = False
        elif isinstance(expected, (int, bool)):
            matches[key] = (val == expected)
        elif isinstance(expected, str):
            matches[key] = (str(val).strip().upper() == expected.strip().upper())
        else:
            matches[key] = (val == expected)
    return matches


class BenchmarkRunner:
    """Orchestrates streaming LLM inference and measures microsecond latencies."""

    def __init__(
        self,
        endpoint: str = "http://localhost:8080/v1",
        model_name: str = "vexel-qwen-2.5-1.5b-q4_k_m",
        api_key: Optional[str] = None,
        concurrency: int = 4,
        timeout_sec: float = 30.0,
        simulate: bool = False,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.concurrency = concurrency
        self.timeout_sec = timeout_sec
        self.simulate = simulate

    @classmethod
    def load_dataset(cls, dataset_path: Union[str, Path]) -> List[BenchmarkSample]:
        """Load benchmark samples from a JSONL file."""
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found: {path.resolve()}")

        samples: List[BenchmarkSample] = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                data = json.loads(line)
                sample_id = data.get("id", f"sample_{idx+1:03d}")
                input_text = data.get("input_text") or data.get("prompt") or data.get("text", "")
                ground_truth = data.get("ground_truth") or data.get("expected", {})
                samples.append(
                    BenchmarkSample(id=str(sample_id), input_text=input_text, ground_truth=ground_truth)
                )

        if not samples:
            raise ValueError(f"No valid benchmark samples found in {path}")
        return samples

    async def _query_model_stream(
        self,
        client: httpx.AsyncClient,
        sample: BenchmarkSample,
        schema_cls: Type[BaseModel],
    ) -> Tuple[str, float, float, int]:
        """Execute streaming request to accurately capture TTFT and total latency.

        Returns:
            (full_text, ttft_ms, total_latency_ms, token_count)
        """
        system_instruction = get_system_instruction(schema_cls)
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": sample.input_text},
            ],
            "temperature": 0.0,
            "max_tokens": 128,
            "stream": True,
            # OpenAI / vLLM response format schema hint
            "response_format": {"type": "json_object"},
        }

        url = f"{self.endpoint}/chat/completions"
        start_time = time.perf_counter()
        ttft_recorded = False
        ttft_ms = 0.0
        full_chunks: List[str] = []
        token_count = 0

        async with client.stream("POST", url, headers=headers, json=payload, timeout=self.timeout_sec) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    raw_data = line[len("data: ") :]
                    try:
                        chunk = json.loads(raw_data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if not ttft_recorded:
                                ttft_ms = (time.perf_counter() - start_time) * 1000.0
                                ttft_recorded = True
                            full_chunks.append(content)
                            token_count += 1
                    except Exception:
                        continue

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        if not ttft_recorded:
            ttft_ms = total_latency_ms

        return "".join(full_chunks), ttft_ms, total_latency_ms, max(1, token_count)

    def _simulate_inference(
        self,
        sample: BenchmarkSample,
        schema_cls: Type[BaseModel],
        profile: str,
    ) -> Tuple[str, float, float, int]:
        """Simulate realistic latency and schema emission characteristics."""
        gt = sample.ground_truth.copy()

        if profile == "vexel-fine-tuned":
            # Vexel Quantized SLM: Sub-25ms TTFT, raw JSON, deterministic
            ttft_ms = np.random.normal(loc=19.5, scale=2.8)
            ttft_ms = max(13.2, float(ttft_ms))
            generation_time = np.random.normal(loc=12.0, scale=2.0)
            total_latency_ms = ttft_ms + max(6.0, float(generation_time))
            tokens_generated = random.randint(22, 38)

            # 99.4% schema compliance, perfect raw JSON
            if random.random() < 0.994:
                output_payload = gt
            else:
                # Slight imperfection
                output_payload = gt.copy()
                if "confidence" in output_payload:
                    output_payload["confidence"] = 1.05  # slightly out of bounds
            raw_text = json.dumps(output_payload)

        elif profile == "base-slm":
            # Base Qwen-1.5B without GRPO: higher latency, often markdown fences
            ttft_ms = np.random.normal(loc=46.0, scale=8.0)
            ttft_ms = max(28.0, float(ttft_ms))
            generation_time = np.random.normal(loc=65.0, scale=12.0)
            total_latency_ms = ttft_ms + max(35.0, float(generation_time))
            tokens_generated = random.randint(45, 95)

            # 70% chance of markdown fence hallucination
            output_payload = gt.copy()
            if random.random() < 0.28:
                # Hallucinate extra key or wrong enum
                output_payload["cot_thinking"] = "Analyzed news catalyst."

            serialized = json.dumps(output_payload)
            if random.random() < 0.65:
                raw_text = f"```json\n{serialized}\n```"
            else:
                raw_text = serialized

        else:  # "frontier-api" (e.g. GPT-4o / Claude 3.5 Sonnet over WAN)
            # Frontier: High WAN latency + high TTFT, high reasoning
            ttft_ms = np.random.normal(loc=580.0, scale=95.0)
            ttft_ms = max(380.0, float(ttft_ms))
            generation_time = np.random.normal(loc=520.0, scale=85.0)
            total_latency_ms = ttft_ms + max(280.0, float(generation_time))
            tokens_generated = random.randint(40, 75)

            output_payload = gt.copy()
            serialized = json.dumps(output_payload)
            if random.random() < 0.12:
                raw_text = f"Here is the quantitative payload:\n```json\n{serialized}\n```"
            else:
                raw_text = serialized

        return raw_text, ttft_ms, total_latency_ms, tokens_generated

    async def run_single_sample(
        self,
        client: Optional[httpx.AsyncClient],
        sample: BenchmarkSample,
        schema_cls: Type[BaseModel],
        profile: str = "vexel-fine-tuned",
    ) -> SampleResult:
        """Run inference on one sample, parse, validate, and evaluate field accuracy."""
        if self.simulate or client is None:
            raw_text, ttft_ms, total_ms, token_count = self._simulate_inference(
                sample, schema_cls, profile=profile
            )
        else:
            try:
                raw_text, ttft_ms, total_ms, token_count = await self._query_model_stream(
                    client, sample, schema_cls
                )
            except Exception as e:
                return SampleResult(
                    sample_id=sample.id,
                    prompt=sample.input_text,
                    raw_text="",
                    parsed_json=None,
                    is_valid_json=False,
                    is_raw_json=False,
                    is_schema_compliant=False,
                    validation_error=f"Inference error: {type(e).__name__}: {str(e)}",
                    ttft_ms=0.0,
                    total_latency_ms=0.0,
                    tokens_generated=0,
                    field_matches={k: False for k in sample.ground_truth.keys()},
                    ground_truth=sample.ground_truth,
                )

        parsed_json, is_valid_json, is_raw_json = extract_json_candidate(raw_text)
        is_schema_compliant = False
        val_error: Optional[str] = None
        field_matches: Dict[str, bool] = {}

        if is_valid_json and parsed_json is not None:
            try:
                schema_cls.model_validate(parsed_json)
                is_schema_compliant = True
            except ValidationError as ve:
                val_error = str(ve).split("\n")[0]
            except Exception as e:
                val_error = str(e)

            field_matches = evaluate_field_accuracy(parsed_json, sample.ground_truth)
        else:
            val_error = "Failed to parse JSON candidate"
            field_matches = {k: False for k in sample.ground_truth.keys()}

        return SampleResult(
            sample_id=sample.id,
            prompt=sample.input_text,
            raw_text=raw_text,
            parsed_json=parsed_json,
            is_valid_json=is_valid_json,
            is_raw_json=is_raw_json,
            is_schema_compliant=is_schema_compliant,
            validation_error=val_error,
            ttft_ms=ttft_ms,
            total_latency_ms=total_ms,
            tokens_generated=token_count,
            field_matches=field_matches,
            ground_truth=sample.ground_truth,
        )

    def _aggregate_metrics(
        self,
        results: List[SampleResult],
        effective_model: str,
        schema_cls: Type[BaseModel],
    ) -> ModelBenchmarkMetrics:
        """Aggregate granular sample results into summary metrics."""
        n = len(results)
        valid_json = sum(1 for r in results if r.is_valid_json)
        raw_json = sum(1 for r in results if r.is_raw_json)
        schema_comp = sum(1 for r in results if r.is_schema_compliant)

        ttft_vals = [r.ttft_ms for r in results if r.ttft_ms > 0] or [0.0]
        lat_vals = [r.total_latency_ms for r in results if r.total_latency_ms > 0] or [0.0]
        total_tokens = sum(r.tokens_generated for r in results)
        total_seconds = max(0.001, sum(r.total_latency_ms for r in results) / 1000.0)

        # Field-level accuracy calculation
        field_totals: Dict[str, int] = {}
        field_correct: Dict[str, int] = {}
        for r in results:
            for k, match in r.field_matches.items():
                field_totals[k] = field_totals.get(k, 0) + 1
                if match:
                    field_correct[k] = field_correct.get(k, 0) + 1

        field_accuracies = {
            k: (field_correct.get(k, 0) / field_totals[k]) * 100.0
            for k in field_totals
        }
        overall_acc = (
            sum(field_correct.values()) / max(1, sum(field_totals.values()))
        ) * 100.0

        return ModelBenchmarkMetrics(
            model_name=effective_model,
            schema_name=schema_cls.__name__,
            total_samples=n,
            valid_json_count=valid_json,
            raw_json_count=raw_json,
            valid_json_rate_pct=(valid_json / n) * 100.0 if n else 0.0,
            raw_json_rate_pct=(raw_json / n) * 100.0 if n else 0.0,
            schema_compliant_count=schema_comp,
            schema_compliance_rate_pct=(schema_comp / n) * 100.0 if n else 0.0,
            ttft_p50_ms=float(np.percentile(ttft_vals, 50)),
            ttft_p95_ms=float(np.percentile(ttft_vals, 95)),
            ttft_p99_ms=float(np.percentile(ttft_vals, 99)),
            latency_p50_ms=float(np.percentile(lat_vals, 50)),
            latency_p95_ms=float(np.percentile(lat_vals, 95)),
            latency_p99_ms=float(np.percentile(lat_vals, 99)),
            field_accuracies_pct=field_accuracies,
            overall_field_accuracy_pct=overall_acc,
            avg_tokens_per_sec=float(total_tokens / total_seconds),
            results=results,
        )

    def run_single_sample_sync(
        self,
        sample: BenchmarkSample,
        schema_cls: Type[BaseModel],
        profile: str = "vexel-fine-tuned",
    ) -> SampleResult:
        """Synchronous single-sample evaluation, ideal for offline simulation."""
        raw_text, ttft_ms, total_ms, token_count = self._simulate_inference(
            sample, schema_cls, profile=profile
        )
        parsed_json, is_valid_json, is_raw_json = extract_json_candidate(raw_text)
        is_schema_compliant = False
        val_error: Optional[str] = None
        field_matches: Dict[str, bool] = {}

        if is_valid_json and parsed_json is not None:
            try:
                schema_cls.model_validate(parsed_json)
                is_schema_compliant = True
            except ValidationError as ve:
                val_error = str(ve).split("\n")[0]
            except Exception as e:
                val_error = str(e)

            field_matches = evaluate_field_accuracy(parsed_json, sample.ground_truth)
        else:
            val_error = "Failed to parse JSON candidate"
            field_matches = {k: False for k in sample.ground_truth.keys()}

        return SampleResult(
            sample_id=sample.id,
            prompt=sample.input_text,
            raw_text=raw_text,
            parsed_json=parsed_json,
            is_valid_json=is_valid_json,
            is_raw_json=is_raw_json,
            is_schema_compliant=is_schema_compliant,
            validation_error=val_error,
            ttft_ms=ttft_ms,
            total_latency_ms=total_ms,
            tokens_generated=token_count,
            field_matches=field_matches,
            ground_truth=sample.ground_truth,
        )

    def run_benchmark_sync(
        self,
        samples: List[BenchmarkSample],
        schema_cls: Type[BaseModel],
        profile: str = "vexel-fine-tuned",
        custom_model_name: Optional[str] = None,
    ) -> ModelBenchmarkMetrics:
        """Synchronous benchmark execution for simulation and offline analysis."""
        effective_model = custom_model_name or self.model_name
        results = [
            self.run_single_sample_sync(s, schema_cls, profile=profile)
            for s in samples
        ]
        return self._aggregate_metrics(results, effective_model, schema_cls)

    async def run_benchmark(
        self,
        samples: List[BenchmarkSample],
        schema_cls: Type[BaseModel],
        profile: str = "vexel-fine-tuned",
        custom_model_name: Optional[str] = None,
    ) -> ModelBenchmarkMetrics:
        """Run all benchmark samples with bounded concurrency."""
        sem = asyncio.Semaphore(self.concurrency)
        effective_model = custom_model_name or self.model_name

        async def worker(
            client: Optional[httpx.AsyncClient], s: BenchmarkSample
        ) -> SampleResult:
            async with sem:
                return await self.run_single_sample(
                    client=client, sample=s, schema_cls=schema_cls, profile=profile
                )

        results: List[SampleResult] = []
        if self.simulate:
            tasks = [worker(None, s) for s in samples]
            results = await asyncio.gather(*tasks)
        else:
            limits = httpx.Limits(max_keepalive_connections=self.concurrency, max_connections=self.concurrency * 2)
            async with httpx.AsyncClient(limits=limits, timeout=self.timeout_sec) as client:
                tasks = [worker(client, s) for s in samples]
                results = await asyncio.gather(*tasks)

        return self._aggregate_metrics(results, effective_model, schema_cls)
